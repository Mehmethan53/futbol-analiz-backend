"""Seçilen maçın puan durumu, son 5 maçı, formdaki oyuncuları ve eksikleri.

Sayılar ESPN fikstüründen hesaplanır. Kaynakta olmayan sakat veya
golcü uydurulmaz.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from pydantic import BaseModel, Field

from bulten import BultenMac, mac_baglami

_SURE = 180
_kilit = threading.Lock()
_onbellek: dict[str, tuple[float, dict]] = {}
_YOK_DURUM = {
    "out": "kadro dışı",
    "injured": "sakat",
    "injury": "sakat",
    "suspended": "cezalı",
    "suspension": "cezalı",
    "doubtful": "şüpheli",
    "questionable": "şüpheli",
}


class SonBes(BaseModel):
    mac_sayisi: int = 0
    atilan_ortalama: float = 0
    yenilen_ortalama: float = 0
    ust_2_5: int = 0
    ust_3_5: int = 0
    maclar: list[str] = Field(default_factory=list)


class DerinBaglam(BaseModel):
    gemini_metni: str = ""
    ev_son_bes: SonBes = Field(default_factory=SonBes)
    dep_son_bes: SonBes = Field(default_factory=SonBes)
    ev_formdakiler: list[str] = Field(default_factory=list)
    dep_formdakiler: list[str] = Field(default_factory=list)
    ev_eksikler: list[str] = Field(default_factory=list)
    dep_eksikler: list[str] = Field(default_factory=list)
    puan_durumu: str = ""


def derin_baglam(mac: BultenMac) -> DerinBaglam:
    try:
        return _derin_baglam(mac)
    except Exception:
        bos = "Bu maç için derin istatistik alınamadı. Eksik isim uydurma."
        return DerinBaglam(gemini_metni=f"{mac_baglami(mac)}\n{bos}")


def _derin_baglam(mac: BultenMac) -> DerinBaglam:
    ozet = _json_al(f"{mac.lig_kodu}/summary?event={mac.etkinlik_id}") if mac.etkinlik_id else {}
    ev_bes, ev_idler = _son_bes(ozet.get("lastFiveGames"), mac.ev_takim_id)
    dep_bes, dep_idler = _son_bes(ozet.get("lastFiveGames"), mac.dep_takim_id)
    ev_form = _katkilar(mac.lig_kodu, mac.ev_takim_id, ev_idler)
    dep_form = _katkilar(mac.lig_kodu, mac.dep_takim_id, dep_idler)
    if not ev_form:
        ev_form = _sezon_liderleri(ozet.get("leaders"), mac.ev_takim_id)
    if not dep_form:
        dep_form = _sezon_liderleri(ozet.get("leaders"), mac.dep_takim_id)
    ev_eksik = _eksikler(mac.lig_kodu, mac.ev_takim_id)
    dep_eksik = _eksikler(mac.lig_kodu, mac.dep_takim_id)
    puan = _puan_durumu(ozet.get("standings"), mac)
    metin = _gemini_metni(mac, ev_bes, dep_bes, ev_form, dep_form, ev_eksik, dep_eksik, puan)
    return DerinBaglam(
        gemini_metni=metin,
        ev_son_bes=ev_bes,
        dep_son_bes=dep_bes,
        ev_formdakiler=ev_form,
        dep_formdakiler=dep_form,
        ev_eksikler=ev_eksik,
        dep_eksikler=dep_eksik,
        puan_durumu=puan,
    )


def _son_bes(last_five: object, takim_id: str) -> tuple[SonBes, list[str]]:
    blok = _takim_blogu(last_five, takim_id)
    if not blok:
        return SonBes(), []
    olaylar = [olay for olay in (blok.get("events") or []) if isinstance(olay, dict)]
    olaylar.sort(key=lambda olay: str(olay.get("gameDate") or ""), reverse=True)
    olaylar = olaylar[:5]
    atilan = yenilen = ust25 = ust35 = 0
    kullanilan = 0
    satirlar: list[str] = []
    kimlikler: list[str] = []
    for olay in olaylar:
        kimlik = str(olay.get("id") or "")
        if kimlik:
            kimlikler.append(kimlik)
        skorlar = _skor_cifti(olay, takim_id)
        etiket = str(olay.get("score") or "")
        sonuc = {"W": "galibiyet", "D": "beraberlik", "L": "mağlubiyet"}.get(str(olay.get("gameResult") or ""), "")
        if skorlar is None:
            if etiket:
                satirlar.append(f"{etiket} {sonuc}".strip())
            continue
        att, yed = skorlar
        kullanilan += 1
        atilan += att
        yenilen += yed
        toplam = att + yed
        ust25 += int(toplam >= 3)
        ust35 += int(toplam >= 4)
        satirlar.append(f"{etiket or f'{att}-{yed}'} {sonuc}".strip())
    if kullanilan == 0:
        return SonBes(maclar=satirlar[:5]), kimlikler
    return (
        SonBes(
            mac_sayisi=kullanilan,
            atilan_ortalama=round(atilan / kullanilan, 2),
            yenilen_ortalama=round(yenilen / kullanilan, 2),
            ust_2_5=ust25,
            ust_3_5=ust35,
            maclar=satirlar[:5],
        ),
        kimlikler,
    )


def _takim_blogu(last_five: object, takim_id: str) -> Optional[dict]:
    if not isinstance(last_five, list) or not takim_id:
        return None
    for blok in last_five:
        if isinstance(blok, dict) and str((blok.get("team") or {}).get("id") or "") == takim_id:
            return blok
    return None


def _skor_cifti(olay: dict, takim_id: str) -> Optional[tuple[int, int]]:
    try:
        ev = int(float(olay.get("homeTeamScore")))
        dep = int(float(olay.get("awayTeamScore")))
    except (TypeError, ValueError):
        return None
    if str(olay.get("homeTeamId") or "") == takim_id:
        return ev, dep
    if str(olay.get("awayTeamId") or "") == takim_id:
        return dep, ev
    return None


def _katkilar(slug: str, takim_id: str, etkinlikler: list[str]) -> list[str]:
    if not slug or not takim_id or not etkinlikler:
        return []
    sayac: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=5) as havuz:
        ozetler = havuz.map(lambda kimlik: _json_al(f"{slug}/summary?event={kimlik}"), etkinlikler[:5])
    for ozet in ozetler:
        for play in ozet.get("keyEvents") or []:
            if not isinstance(play, dict) or not play.get("scoringPlay") or play.get("shootout"):
                continue
            if str((play.get("team") or {}).get("id") or "") != takim_id:
                continue
            tur = str((play.get("type") or {}).get("text") or "").lower()
            if "own" in tur or ("goal" not in tur and "penalty" not in tur):
                continue
            kisiler = play.get("participants") or []
            if not kisiler or not isinstance(kisiler[0], dict):
                continue
            golcu = str((kisiler[0].get("athlete") or {}).get("displayName") or "").strip()
            if golcu:
                sayac.setdefault(golcu, [0, 0])[0] += 1
            metin = str(play.get("text") or "").lower()
            if "assist" in metin and len(kisiler) > 1 and isinstance(kisiler[1], dict):
                asist = str((kisiler[1].get("athlete") or {}).get("displayName") or "").strip()
                if asist:
                    sayac.setdefault(asist, [0, 0])[1] += 1
    sirali = sorted(sayac.items(), key=lambda oge: (oge[1][0] * 2 + oge[1][1], oge[1][0]), reverse=True)
    return [f"{ad}: son maçlarda {gol} gol, {asist} asist" for ad, (gol, asist) in sirali[:4] if gol or asist]


def _sezon_liderleri(leaders: object, takim_id: str) -> list[str]:
    if not isinstance(leaders, list) or not takim_id:
        return []
    for blok in leaders:
        if not isinstance(blok, dict) or str((blok.get("team") or {}).get("id") or "") != takim_id:
            continue
        satirlar: list[str] = []
        for kategori in blok.get("leaders") or []:
            if not isinstance(kategori, dict):
                continue
            ad = f"{kategori.get('name') or ''} {kategori.get('displayName') or ''}".lower()
            if "goal" not in ad and "assist" not in ad:
                continue
            for kisi in (kategori.get("leaders") or [])[:2]:
                if not isinstance(kisi, dict):
                    continue
                isim = str((kisi.get("athlete") or {}).get("displayName") or "").strip()
                deger = str(kisi.get("shortDisplayValue") or kisi.get("displayValue") or "").strip()
                if isim and deger:
                    satirlar.append(f"{isim}: {deger} (sezon lideri, son 5 maç dökümü yok)")
        return satirlar[:4]
    return []


def _eksikler(slug: str, takim_id: str) -> list[str]:
    if not slug or not takim_id:
        return ["Kadro servisi bu takım için sakat veya cezalı bildirmiyor."]
    kadro = _json_al(f"{slug}/teams/{takim_id}/roster")
    bulunan: list[str] = []
    for oyuncu in _oyuncular(kadro):
        isim = str(oyuncu.get("displayName") or oyuncu.get("fullName") or "").strip()
        if not isim:
            continue
        sakatlik = _sakat_metni(oyuncu.get("injuries") or [])
        durum = oyuncu.get("status") or {}
        durum_ad = str(durum.get("name") or durum.get("type") or "") if isinstance(durum, dict) else str(durum)
        turkce = _YOK_DURUM.get(durum_ad.strip().lower(), "")
        if sakatlik:
            bulunan.append(f"{isim}: {sakatlik}")
        elif turkce:
            bulunan.append(f"{isim}: {turkce}")
        if len(bulunan) == 8:
            break
    return bulunan or ["Kadro servisi bu takım için sakat veya cezalı bildirmiyor."]


def _oyuncular(kadro: dict) -> list[dict]:
    ham = kadro.get("athletes") or []
    if ham and isinstance(ham[0], dict) and "items" in ham[0] and "displayName" not in ham[0]:
        duz: list[dict] = []
        for grup in ham:
            duz.extend(oge for oge in (grup.get("items") or []) if isinstance(oge, dict))
        return duz
    return [oge for oge in ham if isinstance(oge, dict)]


def _sakat_metni(sakatliklar: object) -> str:
    if not isinstance(sakatliklar, list) or not sakatliklar:
        return ""
    sakat = sakatliklar[0]
    if not isinstance(sakat, dict):
        return "sakat"
    parcalar: list[str] = []
    for alan in ("status", "type", "details", "longComment", "shortComment"):
        deger = sakat.get(alan)
        if isinstance(deger, dict):
            deger = deger.get("description") or deger.get("detail") or deger.get("name") or ""
        metin = str(deger or "").strip()
        if metin and metin not in parcalar:
            parcalar.append(metin)
    return " · ".join(parcalar)[:160] or "sakat"


def _puan_durumu(standings: object, mac: BultenMac) -> str:
    if not isinstance(standings, dict):
        return "Puan durumu bu müsabaka için gelmedi."
    hedefler = {mac.ev_takim_id: mac.ev_sahibi, mac.dep_takim_id: mac.deplasman}
    bulunan: list[str] = []
    for grup in standings.get("groups") or []:
        if not isinstance(grup, dict):
            continue
        tablo = grup.get("standings") or {}
        kayitlar = tablo.get("entries") if isinstance(tablo, dict) else None
        if not isinstance(kayitlar, list):
            continue
        grup_adi = str(grup.get("name") or "").strip()
        if not grup_adi:
            baslik = str(grup.get("header") or "")
            if "standing" not in baslik.casefold():
                grup_adi = baslik.strip()
        for kayit in kayitlar:
            if not isinstance(kayit, dict):
                continue
            kimlik = str(kayit.get("id") or "")
            if kimlik not in hedefler:
                continue
            istat = {}
            for satir in kayit.get("stats") or []:
                if isinstance(satir, dict) and satir.get("name"):
                    istat[str(satir["name"])] = str(satir.get("displayValue") or "")
            sira = istat.get("rank", "?")
            puan = istat.get("points", "?")
            averaj = istat.get("pointDifferential", "?")
            on_ek = f"{grup_adi}: " if grup_adi else ""
            bulunan.append(
                f"{on_ek}{hedefler[kimlik]} {sira}. sıra, {puan} puan, averaj {averaj}, "
                f"{istat.get('wins', '?')} galibiyet, {istat.get('ties', '?')} beraberlik, "
                f"{istat.get('losses', '?')} mağlubiyet"
            )
    return " | ".join(bulunan) if bulunan else "Puan durumu bu müsabaka için gelmedi."


def _gemini_metni(
    mac: BultenMac,
    ev_bes: SonBes,
    dep_bes: SonBes,
    ev_form: list[str],
    dep_form: list[str],
    ev_eksik: list[str],
    dep_eksik: list[str],
    puan: str,
) -> str:
    def blok(ad: str, form: SonBes, katki: list[str], eksik: list[str]) -> str:
        if form.mac_sayisi:
            istat = (
                f"Son {form.mac_sayisi} resmi maç: attığı gol ortalaması {form.atilan_ortalama}, "
                f"yediği gol ortalaması {form.yenilen_ortalama}. "
                f"2.5 üst biten {form.ust_2_5}/{form.mac_sayisi}, "
                f"3.5 üst biten {form.ust_3_5}/{form.mac_sayisi}. "
                f"Maçlar: {'; '.join(form.maclar)}."
            )
        else:
            istat = "Son 5 maçın gol dökümü bu servisten gelmedi."
        return (
            f"{ad}. {istat} "
            f"Form katkısı: {'; '.join(katki) if katki else 'isimli gol/asist dökümü yok'}. "
            f"Sakat veya cezalı: {'; '.join(eksik)}."
        )

    return "\n".join(
        [
            mac_baglami(mac),
            "Aşağıdaki sayılar ve isimler dış kaynaktan geldi. Listede olmayan oyuncu, sakatlık veya skor uydurma.",
            f"Puan durumu: {puan}",
            blok(mac.ev_sahibi, ev_bes, ev_form, ev_eksik),
            blok(mac.deplasman, dep_bes, dep_form, dep_eksik),
            "Özette bu dört başlığı ayrı satırlar halinde kullan: puan durumu, son 5 maç ve 2.5/3.5 üst sayıları, "
            "formdaki oyuncular, sakat veya cezalılar. 1X2 ve alt/üst olasılıklarını bu sayılara dayandır. "
            "Kupon, oran ya da 'oyna' ifadesi yazma.",
        ]
    )


def _json_al(yol: str) -> dict:
    anahtar = yol
    simdi = time.time()
    with _kilit:
        kayit = _onbellek.get(anahtar)
        if kayit and simdi - kayit[0] < _SURE:
            return kayit[1]
    adres = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{yol}"
    try:
        istek = urllib.request.Request(adres, headers={"User-Agent": "IddaaAIAnalysis/0.3"})
        with urllib.request.urlopen(istek, timeout=12) as yanit:
            veri = json.load(yanit)
        if not isinstance(veri, dict):
            veri = {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        veri = {}
    with _kilit:
        _onbellek[anahtar] = (time.time(), veri)
    return veri
