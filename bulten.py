"""Güncel futbol bülteni.

Maçlar ESPN'in herkese açık skor servisinden gelir. Liste sahte
takımlarla doldurulmaz. Kaynak yanıt vermezse hata döner.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

IST = ZoneInfo("Europe/Istanbul")
AYLAR = ("Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara")
LIGLER = (
    ("uefa.nations", "UEFA Uluslar Ligi"),
    ("caf.nations", "Afrika Uluslar Kupası"),
    ("concacaf.nations.league", "CONCACAF Uluslar Ligi"),
    ("fifa.friendly", "Uluslararası Hazırlık"),
    ("uefa.champions", "Şampiyonlar Ligi"),
    ("uefa.wchampions", "Şampiyonlar Ligi (Kadınlar)"),
    ("uefa.europa", "Avrupa Ligi"),
    ("tur.1", "Süper Lig"),
    ("eng.1", "Premier League"),
    ("esp.1", "La Liga"),
    ("ger.1", "Bundesliga"),
    ("ita.1", "Serie A"),
    ("fra.1", "Ligue 1"),
    ("ned.1", "Eredivisie"),
)
TAKIP_EDILEN = {kod for kod, _ad in LIGLER[:6]}
ISTATISTIK_ADLARI = {
    "possessionPct": "topa_sahip",
    "possession": "topa_sahip",
    "shotsTotal": "sut",
    "totalShots": "sut",
    "shots": "sut",
    "shotsOnTarget": "isabet",
    "shotsOnGoal": "isabet",
    "wonCorners": "korner",
    "cornerKicks": "korner",
    "foulsCommitted": "faul",
    "fouls": "faul",
}
_ONBELLEK_SANIYE = 45
_kilit = threading.Lock()
_onbellek: Optional[tuple[float, "Bulten"]] = None


class BultenHatasi(Exception):
    def __init__(self, mesaj: str, durum_kodu: int = 503) -> None:
        super().__init__(mesaj)
        self.durum_kodu = durum_kodu


class SahaIstatistigi(BaseModel):
    dakika: int = Field(ge=0, le=120)
    ev_skor: int = Field(ge=0, le=30)
    dep_skor: int = Field(ge=0, le=30)
    ev_sut: int = Field(ge=0)
    dep_sut: int = Field(ge=0)
    ev_isabetli_sut: int = Field(ge=0)
    dep_isabetli_sut: int = Field(ge=0)
    ev_topa_sahip: int = Field(ge=0, le=100)
    dep_topa_sahip: int = Field(ge=0, le=100)
    ev_korner: int = Field(ge=0)
    dep_korner: int = Field(ge=0)
    ev_faul: int = Field(ge=0)
    dep_faul: int = Field(ge=0)


class BultenMac(BaseModel):
    id: str
    ev_sahibi: str
    deplasman: str
    lig: str
    baslangic: str
    saat_etiketi: str
    durum: str
    istatistik: Optional[SahaIstatistigi] = None
    lig_kodu: str = ""
    etkinlik_id: str = ""
    ev_takim_id: str = ""
    dep_takim_id: str = ""


class Bulten(BaseModel):
    kaynak: str
    uyari: str
    canli: list[BultenMac]
    yaklasan: list[BultenMac]


def bulteni_getir() -> Bulten:
    global _onbellek
    simdi = time.time()
    with _kilit:
        if _onbellek and simdi - _onbellek[0] < _ONBELLEK_SANIYE:
            return _onbellek[1]
    bulten = _uzaktan_cek()
    with _kilit:
        _onbellek = (time.time(), bulten)
    return bulten


def mac_bul(mac_id: str) -> Optional[BultenMac]:
    bulten = bulteni_getir()
    for mac in (*bulten.canli, *bulten.yaklasan):
        if mac.id == mac_id:
            return mac
    return None


def mac_baglami(mac: BultenMac) -> str:
    if mac.istatistik is None:
        return (
            f"{mac.saat_etiketi} başlayacak {mac.lig} maçı. "
            "Güncel kadro bu akışta yok. Emin olmadığın istatistiği uydurma."
        )
    saha = mac.istatistik
    return (
        "Skor ESPN skor servisinden geldi. "
        f"Dakika {saha.dakika}, skor {saha.ev_skor}-{saha.dep_skor}. "
        f"Şut {saha.ev_sut}-{saha.dep_sut}, isabetli şut "
        f"{saha.ev_isabetli_sut}-{saha.dep_isabetli_sut}. "
        f"Topla oynama %{saha.ev_topa_sahip}-%{saha.dep_topa_sahip}. "
        f"Korner {saha.ev_korner}-{saha.dep_korner}. "
        f"Faul {saha.ev_faul}-{saha.dep_faul}. "
        "Akışta sıfır görünen bir istatistik gelmemiş olabilir; onu kesin veri sayma."
    )


def _uzaktan_cek() -> Bulten:
    bugun = datetime.now(IST).date()
    gunler = [bugun + timedelta(days=i) for i in range(12)]
    bulunan: dict[str, BultenMac] = {}
    hatalar = 0
    isler = [(slug, lig, gun) for slug, lig in LIGLER for gun in gunler]
    with ThreadPoolExecutor(max_workers=16) as havuz:
        gorevler = [havuz.submit(_guvenli_cek, *is_) for is_ in isler]
        for gorev in as_completed(gorevler):
            maclar, hata = gorev.result()
            hatalar += int(hata)
            for mac in maclar:
                bulunan[mac.id] = mac
    if not bulunan and hatalar:
        raise BultenHatasi("Güncel maç bülteni şu an alınamadı. Biraz sonra yeniden deneyin.")
    canli = [mac for mac in bulunan.values() if mac.durum == "canli"]
    yaklasan = [mac for mac in bulunan.values() if mac.durum == "yaklasan"]
    canli.sort(key=lambda mac: mac.istatistik.dakika if mac.istatistik else 0, reverse=True)
    yaklasan.sort(key=lambda mac: mac.baslangic)
    uyari = (
        "Bülten ESPN skor servisinden geliyor: Uluslar Ligi, Afrika Uluslar Kupası, "
        "CONCACAF Uluslar Ligi, hazırlık maçları, erkekler ve kadınlar Şampiyonlar Ligi "
        "ile büyük ligler. Resmi İddaa kuponu değildir."
    )
    gelen = {mac.lig_kodu for mac in bulunan.values()}
    bos = [ad for kod, ad in LIGLER if kod in TAKIP_EDILEN and kod not in gelen]
    if bos:
        uyari += " Bu pencerede fikstürü görünmeyenler: " + ", ".join(bos) + "."
    if hatalar:
        uyari += " Bazı ligler bu turda yanıt vermedi."
    if not canli and not yaklasan:
        uyari += " Önümüzdeki 12 günde listelenecek maç görünmüyor."
    return Bulten(kaynak="espn", uyari=uyari, canli=canli, yaklasan=yaklasan)


def _guvenli_cek(slug: str, lig: str, gun: datetime.date) -> tuple[list[BultenMac], bool]:
    try:
        return _lig_gunu(slug, lig, gun), False
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return [], True


def _lig_gunu(slug: str, lig: str, gun: datetime.date) -> list[BultenMac]:
    adres = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/"
        f"{slug}/scoreboard?dates={gun:%Y%m%d}"
    )
    istek = urllib.request.Request(adres, headers={"User-Agent": "IddaaAIAnalysis/0.2"})
    with urllib.request.urlopen(istek, timeout=12) as yanit:
        veri = json.load(yanit)
    maclar: list[BultenMac] = []
    for olay in veri.get("events") or []:
        mac = _olayi_cevir(olay, lig, slug)
        if mac is not None:
            maclar.append(mac)
    return maclar


def _lig_adi(lig: str, yarisma: dict) -> str:
    grup = yarisma.get("grouping") or yarisma.get("group") or {}
    if isinstance(grup, dict):
        ad = str(grup.get("name") or grup.get("shortName") or "").strip()
        if ad and ad.casefold() not in lig.casefold():
            return f"{lig} · {ad}"
    for not_ in yarisma.get("notes") or []:
        if not isinstance(not_, dict):
            continue
        baslik = str(not_.get("headline") or "").strip()
        kucuk = baslik.casefold()
        if baslik and ("group" in kucuk or "grup" in kucuk) and baslik.casefold() not in lig.casefold():
            return f"{lig} · {baslik}"
    return lig


def _olayi_cevir(olay: dict, lig: str, slug: str) -> Optional[BultenMac]:
    kimlik = str(olay.get("id") or "").strip()
    yarismalar = olay.get("competitions") or []
    if not kimlik or not yarismalar:
        return None
    yarisma = yarismalar[0]
    durum = yarisma.get("status") or {}
    tur = durum.get("type") or {}
    hal = str(tur.get("state") or "")
    aciklama = f"{tur.get('description') or ''} {tur.get('shortDetail') or ''}".lower()
    if hal == "post" or "postpon" in aciklama:
        return None
    rakipler = {str(taraf.get("homeAway")): taraf for taraf in yarisma.get("competitors") or []}
    ev = rakipler.get("home")
    dep = rakipler.get("away")
    if not ev or not dep:
        return None
    ev_adi = str((ev.get("team") or {}).get("displayName") or "").strip()
    dep_adi = str((dep.get("team") or {}).get("displayName") or "").strip()
    if len(ev_adi) < 2 or len(dep_adi) < 2:
        return None
    baslangic = _istanbul(str(olay.get("date") or ""))
    saat = _saat_etiketi(baslangic)
    ortak = {
        "id": f"{slug}-{kimlik}",
        "ev_sahibi": ev_adi,
        "deplasman": dep_adi,
        "lig": _lig_adi(lig, yarisma),
        "lig_kodu": slug,
        "etkinlik_id": kimlik,
        "ev_takim_id": str((ev.get("team") or {}).get("id") or ""),
        "dep_takim_id": str((dep.get("team") or {}).get("id") or ""),
    }
    if hal == "in":
        return BultenMac(
            **ortak,
            baslangic=baslangic.isoformat() if baslangic else "",
            saat_etiketi="Canlı",
            durum="canli",
            istatistik=_saha(ev, dep, durum),
        )
    if baslangic is None:
        return None
    return BultenMac(**ortak, baslangic=baslangic.isoformat(), saat_etiketi=saat, durum="yaklasan")


def _saha(ev: dict, dep: dict, durum: dict) -> SahaIstatistigi:
    ev_sayi = _istatistikler(ev)
    dep_sayi = _istatistikler(dep)
    ev_sahip = ev_sayi.get("topa_sahip", 0)
    dep_sahip = dep_sayi.get("topa_sahip", 0)
    if ev_sahip and not dep_sahip:
        dep_sahip = max(0, 100 - ev_sahip)
    if dep_sahip and not ev_sahip:
        ev_sahip = max(0, 100 - dep_sahip)
    return SahaIstatistigi(
        dakika=_dakika(str(durum.get("displayClock") or "")),
        ev_skor=_skor(ev.get("score")),
        dep_skor=_skor(dep.get("score")),
        ev_sut=ev_sayi.get("sut", 0),
        dep_sut=dep_sayi.get("sut", 0),
        ev_isabetli_sut=ev_sayi.get("isabet", 0),
        dep_isabetli_sut=dep_sayi.get("isabet", 0),
        ev_topa_sahip=min(ev_sahip, 100),
        dep_topa_sahip=min(dep_sahip, 100),
        ev_korner=ev_sayi.get("korner", 0),
        dep_korner=dep_sayi.get("korner", 0),
        ev_faul=ev_sayi.get("faul", 0),
        dep_faul=dep_sayi.get("faul", 0),
    )


def _istatistikler(taraf: dict) -> dict[str, int]:
    sonuc: dict[str, int] = {}
    for satir in taraf.get("statistics") or []:
        alan = ISTATISTIK_ADLARI.get(str(satir.get("name") or ""))
        if not alan:
            continue
        sonuc[alan] = _skor(satir.get("displayValue"))
    return sonuc


def _dakika(saat: str) -> int:
    rakam = ""
    for karakter in saat:
        if karakter.isdigit():
            rakam += karakter
        elif rakam:
            break
    if not rakam:
        return 45 if "HT" in saat.upper() else 1
    return min(int(rakam), 120)


def _skor(deger: object) -> int:
    metin = str(deger or "0").replace("%", "").strip()
    try:
        return max(0, int(float(metin)))
    except ValueError:
        return 0


def _istanbul(metin: str) -> Optional[datetime]:
    if not metin:
        return None
    try:
        an = datetime.fromisoformat(metin.replace("Z", "+00:00"))
    except ValueError:
        return None
    if an.tzinfo is None:
        an = an.replace(tzinfo=IST)
    return an.astimezone(IST)


def _saat_etiketi(an: Optional[datetime]) -> str:
    if an is None:
        return "Tarih yok"
    return f"{an.day} {AYLAR[an.month - 1]} · {an:%H:%M}"
