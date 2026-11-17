"""Maç sonucu, alt/üst, korner ve dakikalık gol analizi.

Google Gemini'den yapılandırılmış JSON alır, market olasılıklarını
birbiriyle tutarlı hale getirir ve mobil şemayı üretir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from mac_verisi import DerinBaglam, SonBes

load_dotenv(Path(__file__).resolve().parent / ".env")
MODEL_ADI = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"

SABIT_ARALIKLAR: tuple[tuple[int, int, str], ...] = (
    (1, 15, "1-15"),
    (16, 30, "16-30"),
    (31, 45, "31-45"),
    (46, 60, "46-60"),
    (61, 75, "61-75"),
    (76, 90, "76-90"),
)

UYARI = (
    "Bu çıktı istatistiksel bir ön değerlendirmedir; bahis tavsiyesi, "
    "kupon veya kazanç vaadi değildir. Sonuç garantisi yoktur. "
    "18 yaşından küçükler bahis oynayamaz."
)

SISTEM_MESAJI = """
Sen profesyonel bir futbol maç analistsin. Seçilen maç için aşağıdaki
pazarların tamamını olasılık yüzdesi ve kısa gerekçeyle değerlendir.
Yanıtın yalnızca geçerli bir JSON nesnesi olsun.

Kurallar:
- Bahis tavsiyesi, kupon, "oyna" veya kazanç vaadi yazma. Yalnızca olasılık üret.
- Emin olmadığın kadro, sakatlık veya istatistik uydurma. Veri zayıfsa
  özette söyle ve güven skorunu düşür. Canlı skor verilmişse onu dikkate al.
- Olasılıklar 0 ile 100 arasında yüzde olsun. 0.28 gibi kesir yazma.
- ms_1x2 içindeki 1, X ve 2 toplamı 100 olsun.
- alt_ust_2_5, alt_ust_3_5, kg, iy_1x2, iy_alt_ust_0_5, iy_alt_ust_1_5,
  korner_8_5 ve korner_9_5 içinde üst/alt veya var/yok ya da 1/X/2
  kendi içinde 100 etsin.
- 3.5 üst, 2.5 üstten büyük olmasın. 9.5 korner üst, 8.5 üstten büyük olmasın.
  İlk yarı 1.5 üst, ilk yarı 0.5 üstten büyük olmasın.
- Çifte şansı sen hesaplama; sunucu 1X2'den türetecek.
- Aralıklar tam olarak 1-15, 16-30, 31-45, 46-60, 61-75, 76-90 olsun.
- gol_olasiligi: o dilimde en az bir gol olma olasılığı. Toplamı 100 olmak zorunda değildir.
- ilk_gol_olasiligi değerleri ile golsuz_olasiligi birlikte 100 etsin.
- Metinler Türkçe olsun.
- Sana verilen puan durumu, son 5 maç, 2.5/3.5 üst sayıları, form ve eksik listesi
  dış kaynaktan hesaplanmıştır. Özette bunları ayrı satırlar halinde kullan.
  Listede olmayan oyuncu veya sakatlık adı yazma. Veri yoksa "gelmedi" de.
- ozet, bahis pazarına uygun istatistiksel bir maç raporudur. Kupon, oran
  ya da "oyna" ifadesi yazma.

JSON şeması:
{
  "ozet": "Puan durumu, son 5 maç, form ve eksikler ayrı satırlarda",
  "beklenen_toplam_gol": 2.4,
  "beklenen_korner": 9.4,
  "guven_skoru": 42,
  "golsuz_olasiligi": 12.0,
  "ms_1x2": {"1": 46, "X": 28, "2": 26, "gerekce": "Kısa gerekçe"},
  "alt_ust_2_5": {"ust": 57, "alt": 43, "gerekce": "Kısa gerekçe"},
  "alt_ust_3_5": {"ust": 31, "alt": 69, "gerekce": "Kısa gerekçe"},
  "kg": {"var": 54, "yok": 46, "gerekce": "Kısa gerekçe"},
  "iy_1x2": {"1": 38, "X": 40, "2": 22, "gerekce": "Kısa gerekçe"},
  "iy_alt_ust_0_5": {"ust": 72, "alt": 28, "gerekce": "Kısa gerekçe"},
  "iy_alt_ust_1_5": {"ust": 34, "alt": 66, "gerekce": "Kısa gerekçe"},
  "korner_8_5": {"ust": 55, "alt": 45, "gerekce": "Kısa gerekçe"},
  "korner_9_5": {"ust": 42, "alt": 58, "gerekce": "Kısa gerekçe"},
  "araliklar": [
    {"etiket": "1-15", "baslangic": 1, "bitis": 15, "gol_olasiligi": 28, "ilk_gol_olasiligi": 18, "yorum": "Kısa gerekçe"}
  ],
  "anahtar_faktorler": ["Kısa faktör"],
  "riskler": ["Belirsizlik"]
}
""".strip()


class AnalizHatasi(Exception):
    """Kullanıcıya gösterilebilecek, durum kodu taşıyan analiz hatası."""

    def __init__(self, mesaj: str, durum_kodu: int = 502) -> None:
        super().__init__(mesaj)
        self.durum_kodu = durum_kodu


class GolAraligi(BaseModel):
    etiket: str
    baslangic: int
    bitis: int
    gol_olasiligi: float = Field(ge=0, le=100)
    ilk_gol_olasiligi: float = Field(ge=0, le=100)
    yorum: str


class Secenek(BaseModel):
    kod: str
    etiket: str
    olasilik: float = Field(ge=0, le=100)


class Market(BaseModel):
    kod: str
    baslik: str
    aciklama: str
    gerekce: str
    secenekler: list[Secenek]


class MacAnalizi(BaseModel):
    ev_sahibi: str
    deplasman: str
    lig: str
    tarih: Optional[str] = None
    ozet: str
    beklenen_toplam_gol: float = Field(ge=0, le=12)
    beklenen_korner: float = Field(ge=0, le=25)
    guven_skoru: int = Field(ge=0, le=100)
    golsuz_olasiligi: float = Field(ge=0, le=100)
    araliklar: list[GolAraligi] = Field(min_length=6, max_length=6)
    marketler: list[Market]
    anahtar_faktorler: list[str]
    riskler: list[str]
    kaynak: str
    olasiliklar_normalize_edildi: bool = False
    uyari: str
    ev_son_bes: SonBes = Field(default_factory=SonBes)
    dep_son_bes: SonBes = Field(default_factory=SonBes)
    ev_formdakiler: list[str] = Field(default_factory=list)
    dep_formdakiler: list[str] = Field(default_factory=list)
    ev_eksikler: list[str] = Field(default_factory=list)
    dep_eksikler: list[str] = Field(default_factory=list)
    puan_durumu: str = ""


class AnalizMotoru:
    """Maç bilgisini Gemini analizine çevirir."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        temiz = (api_key or "").strip()
        self.api_key = temiz or None
        self._istemci: Optional[genai.Client] = None

    @property
    def anahtar_var(self) -> bool:
        return self.api_key is not None

    def analiz_et(
        self,
        *,
        ev_sahibi: str,
        deplasman: str,
        lig: str,
        tarih: Optional[str],
        notlar: Optional[str],
        baglam: Optional[DerinBaglam] = None,
    ) -> MacAnalizi:
        if not self.api_key:
            raise AnalizHatasi(
                "GEMINI_API_KEY tanımlı değil. backend/.env dosyasına anahtarı ekleyin.",
                durum_kodu=503,
            )
        ham = self._gemini_cagir(ev_sahibi, deplasman, lig, tarih, notlar)
        return _hamdan_analiz(
            ham,
            ev_sahibi=ev_sahibi,
            deplasman=deplasman,
            lig=lig,
            tarih=tarih,
            kaynak=MODEL_ADI,
            bos_ozet="Model bu maç için yazılı bir özet döndürmedi.",
            baglam=baglam,
        )

    def _gemini_cagir(
        self,
        ev_sahibi: str,
        deplasman: str,
        lig: str,
        tarih: Optional[str],
        notlar: Optional[str],
    ) -> dict:
        satirlar = [
            f"Ev sahibi: {ev_sahibi}",
            f"Deplasman: {deplasman}",
            f"Lig: {lig}",
        ]
        if tarih:
            satirlar.append(f"Tarih: {tarih}")
        if notlar:
            satirlar.append(f"Ek bağlam: {notlar}")
        satirlar.append(
            "Bu maç için 1X2, alt/üst, karşılıklı gol, ilk yarı, korner "
            "ve dakika bazlı gol aralığı analizini JSON olarak üret."
        )
        try:
            cevap = self._istemci_al().models.generate_content(
                model=MODEL_ADI,
                contents="\n".join(satirlar),
                config=types.GenerateContentConfig(
                    system_instruction=SISTEM_MESAJI,
                    temperature=0.3,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW,
                    ),
                ),
            )
        except Exception as exc:
            raise _gemini_hatasi(exc, self.api_key) from exc
        metin = getattr(cevap, "text", None)
        if not metin:
            raise AnalizHatasi("Gemini boş bir yanıt döndürdü.")
        return _json_coz(metin)

    def _istemci_al(self) -> genai.Client:
        if self._istemci is None:
            self._istemci = genai.Client(api_key=self.api_key)
        return self._istemci


def _gemini_hatasi(exc: Exception, api_key: Optional[str]) -> AnalizHatasi:
    metin = str(exc)
    if api_key:
        metin = metin.replace(api_key, "[anahtar]")
    kod = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if kod == 429 or "RESOURCE_EXHAUSTED" in metin or "429" in metin:
        return AnalizHatasi("Gemini istek limiti doldu. Bir süre sonra yeniden deneyin.", 429)
    if kod in (401, 403) or "API_KEY_INVALID" in metin or "PERMISSION_DENIED" in metin:
        return AnalizHatasi("Gemini anahtarı kabul edilmedi. GEMINI_API_KEY değerini kontrol edin.", 401)
    if "timed out" in metin.lower() or "timeout" in metin.lower():
        return AnalizHatasi("Gemini yanıt vermeden zaman aşımına uğradı.", 503)
    if kod == 503 or "UNAVAILABLE" in metin:
        return AnalizHatasi("Gemini şu an yoğun. Bir süre sonra yeniden deneyin.", 503)
    kisa = metin.replace("\n", " ")[:180]
    return AnalizHatasi(f"Gemini isteği başarısız oldu: {kisa}", 502)


def _hamdan_analiz(
    ham: dict,
    *,
    ev_sahibi: str,
    deplasman: str,
    lig: str,
    tarih: Optional[str],
    kaynak: str,
    bos_ozet: str,
    guven_zorla: Optional[int] = None,
    baglam: Optional[DerinBaglam] = None,
) -> MacAnalizi:
    araliklar = _araliklari_duzelt(ham.get("araliklar"))
    araliklar, golsuz, aralik_normalize = _ilk_golu_normalize(
        araliklar,
        _sinirla(_sayi(ham.get("golsuz_olasiligi"), 0), 0, 100),
    )
    marketler, market_normalize = _marketleri_olustur(ham)
    ozet = str(ham.get("ozet") or "").strip() or bos_ozet
    guven = guven_zorla
    if guven is None:
        guven = int(round(_sinirla(_sayi(ham.get("guven_skoru"), 30), 0, 100)))
    return MacAnalizi(
        ev_sahibi=ev_sahibi,
        deplasman=deplasman,
        lig=lig,
        tarih=tarih,
        beklenen_toplam_gol=round(_sinirla(_sayi(ham.get("beklenen_toplam_gol"), 2.4), 0, 12), 2),
        beklenen_korner=round(_sinirla(_sayi(ham.get("beklenen_korner"), 9.0), 0, 25), 1),
        guven_skoru=guven,
        golsuz_olasiligi=golsuz,
        araliklar=araliklar,
        marketler=marketler,
        anahtar_faktorler=_metin_listesi(ham.get("anahtar_faktorler")) or ["Ek faktör belirtilmedi."],
        riskler=_metin_listesi(ham.get("riskler")) or ["Ayrı bir risk notu belirtilmedi."],
        kaynak=kaynak,
        olasiliklar_normalize_edildi=aralik_normalize or market_normalize,
        uyari=UYARI,
        ev_son_bes=baglam.ev_son_bes if baglam else SonBes(),
        dep_son_bes=baglam.dep_son_bes if baglam else SonBes(),
        ev_formdakiler=baglam.ev_formdakiler if baglam else [],
        dep_formdakiler=baglam.dep_formdakiler if baglam else [],
        ev_eksikler=baglam.ev_eksikler if baglam else [],
        dep_eksikler=baglam.dep_eksikler if baglam else [],
        puan_durumu=baglam.puan_durumu if baglam else "",
        ozet=_ozet_metni(ozet, baglam),
    )


def _ozet_metni(ozet: str, baglam: Optional[DerinBaglam]) -> str:
    metin = ozet.strip()
    if baglam is None:
        return metin[:3200]
    ek = _veri_satirlari(baglam)
    if ek:
        metin = f"{ek}\n\n{metin}".strip()
    return metin[:3200]


def _veri_satirlari(baglam: DerinBaglam) -> str:
    def satir(ad: str, form: SonBes) -> str:
        if not form.mac_sayisi:
            return f"{ad} son 5 maç dökümü gelmedi."
        return (
            f"{ad} son {form.mac_sayisi} maç: attığı {form.atilan_ortalama}, yediği {form.yenilen_ortalama}; "
            f"2.5 üst {form.ust_2_5}/{form.mac_sayisi}, 3.5 üst {form.ust_3_5}/{form.mac_sayisi}."
        )

    return "\n".join(
        [
            f"Puan durumu: {baglam.puan_durumu or 'gelmedi.'}",
            satir("Ev sahibi", baglam.ev_son_bes),
            satir("Deplasman", baglam.dep_son_bes),
            "Ev sahibi form: " + ("; ".join(baglam.ev_formdakiler) or "isimli katkı yok."),
            "Deplasman form: " + ("; ".join(baglam.dep_formdakiler) or "isimli katkı yok."),
            "Ev sahibi eksik: " + ("; ".join(baglam.ev_eksikler) or "bildirim yok."),
            "Deplasman eksik: " + ("; ".join(baglam.dep_eksikler) or "bildirim yok."),
        ]
    )


def _marketleri_olustur(ham: dict) -> tuple[list[Market], bool]:
    degisti = False
    ms, ms_gerekce, ms_degisti = _uclu(
        ham.get("ms_1x2"),
        (("1", "Ev sahibi"), ("X", "Beraberlik"), ("2", "Deplasman")),
    )
    iy, iy_gerekce, iy_degisti = _uclu(
        ham.get("iy_1x2"),
        (("1", "İY 1"), ("X", "İY X"), ("2", "İY 2")),
    )
    degisti = ms_degisti or iy_degisti

    ust25, _, g25, d25 = _ikili(ham.get("alt_ust_2_5"))
    ust35, _, g35, d35 = _ikili(ham.get("alt_ust_3_5"))
    ust35, daraldi = _ustu_daralt(ust25, ust35)
    kg_var, _, gkg, dkg = _ikili(ham.get("kg"), sol="var", sag="yok")
    iy05, _, g05, d05 = _ikili(ham.get("iy_alt_ust_0_5"))
    iy15, _, g15, d15 = _ikili(ham.get("iy_alt_ust_1_5"))
    iy15, iy_dar = _ustu_daralt(iy05, iy15)
    k85, _, gk85, dk85 = _ikili(ham.get("korner_8_5"))
    k95, _, gk95, dk95 = _ikili(ham.get("korner_9_5"))
    k95, korner_dar = _ustu_daralt(k85, k95)
    degisti = degisti or d25 or d35 or daraldi or dkg or d05 or d15 or iy_dar or dk85 or dk95 or korner_dar

    marketler = [
        _market("ms_1x2", "Maç sonucu", "1, beraberlik ve 2 birbirini dışlar.", ms_gerekce, [
            ("1", "Ev sahibi", ms[0]),
            ("X", "Beraberlik", ms[1]),
            ("2", "Deplasman", ms[2]),
        ]),
        _market(
            "cifte_sans",
            "Çifte şans",
            "1X2 olasılıklarından türetildi. Seçenekler birbirini dışlamaz.",
            "Ev sahibi, beraberlik ve deplasman payları toplanarak üretildi.",
            [
                ("1X", "1X", round(ms[0] + ms[1], 1)),
                ("12", "12", round(ms[0] + ms[2], 1)),
                ("X2", "X2", round(ms[1] + ms[2], 1)),
            ],
        ),
        _market("alt_ust_2_5", "2.5 Alt / Üst", "Toplam gol çizgisi.", g25, [("ust", "Üst 2.5", ust25), ("alt", "Alt 2.5", round(100 - ust25, 1))]),
        _market("alt_ust_3_5", "3.5 Alt / Üst", "2.5 üstten daha yüksek olamaz.", g35, [("ust", "Üst 3.5", ust35), ("alt", "Alt 3.5", round(100 - ust35, 1))]),
        _market("kg", "Karşılıklı gol", "İki takımın da gol bulması.", gkg, [("var", "KG Var", kg_var), ("yok", "KG Yok", round(100 - kg_var, 1))]),
        _market("iy_1x2", "İlk yarı sonucu", "İlk 45 dakikanın 1X2 dağılımı.", iy_gerekce, [
            ("1", "İY 1", iy[0]),
            ("X", "İY X", iy[1]),
            ("2", "İY 2", iy[2]),
        ]),
        _market("iy_alt_ust_0_5", "İlk yarı 0.5", "İlk yarıda en az bir gol.", g05, [("ust", "İY Üst 0.5", iy05), ("alt", "İY Alt 0.5", round(100 - iy05, 1))]),
        _market("iy_alt_ust_1_5", "İlk yarı 1.5", "0.5 üstten daha yüksek olamaz.", g15, [("ust", "İY Üst 1.5", iy15), ("alt", "İY Alt 1.5", round(100 - iy15, 1))]),
        _market("korner_8_5", "Korner 8.5", "Toplam korner çizgisi.", gk85, [("ust", "Üst 8.5", k85), ("alt", "Alt 8.5", round(100 - k85, 1))]),
        _market("korner_9_5", "Korner 9.5", "8.5 üstten daha yüksek olamaz.", gk95, [("ust", "Üst 9.5", k95), ("alt", "Alt 9.5", round(100 - k95, 1))]),
    ]
    return marketler, degisti


def _market(kod: str, baslik: str, aciklama: str, gerekce: str, secenekler: list[tuple[str, str, float]]) -> Market:
    return Market(
        kod=kod,
        baslik=baslik,
        aciklama=aciklama,
        gerekce=gerekce[:400],
        secenekler=[
            Secenek(kod=kod_s, etiket=etiket, olasilik=round(_sinirla(deger, 0, 100), 1))
            for kod_s, etiket, deger in secenekler
        ],
    )


def _uclu(ham: object, anahtarlar: tuple[tuple[str, str], ...]) -> tuple[list[float], str, bool]:
    veri = ham if isinstance(ham, dict) else {}
    degerler = [_sinirla(_sayi(veri.get(kod)), 0, 100) for kod, _etiket in anahtarlar]
    gerekce = str(veri.get("gerekce") or "").strip() or "Bu market için ayrı gerekçe dönmedi."
    yuvarlak, degisti = _yuzdeye_cek(degerler)
    return yuvarlak, gerekce, degisti


def _ikili(ham: object, sol: str = "ust", sag: str = "alt") -> tuple[float, float, str, bool]:
    veri = ham if isinstance(ham, dict) else {}
    bir = _sinirla(_sayi(veri.get(sol)), 0, 100)
    iki = _sinirla(_sayi(veri.get(sag)), 0, 100)
    gerekce = str(veri.get("gerekce") or "").strip() or "Bu market için ayrı gerekçe dönmedi."
    yuvarlak, degisti = _yuzdeye_cek([bir, iki])
    return yuvarlak[0], yuvarlak[1], gerekce, degisti


def _ustu_daralt(genis: float, dar: float) -> tuple[float, bool]:
    if dar <= genis:
        return dar, False
    return genis, True


def _yuzdeye_cek(degerler: list[float]) -> tuple[list[float], bool]:
    toplam = sum(degerler)
    if toplam <= 0:
        esit = 100 / len(degerler)
        degerler = [esit for _ in degerler]
        degisti = True
    elif 99.0 <= toplam <= 101.0:
        degisti = False
    else:
        degerler = [deger * 100 / toplam for deger in degerler]
        degisti = True
    yuvarlak = [round(_sinirla(deger, 0, 100), 1) for deger in degerler]
    fark = round(100.0 - sum(yuvarlak), 1)
    if fark != 0:
        indeks = max(range(len(yuvarlak)), key=yuvarlak.__getitem__)
        yuvarlak[indeks] = round(_sinirla(yuvarlak[indeks] + fark, 0, 100), 1)
        degisti = True
    return yuvarlak, degisti


def _araliklari_duzelt(ham_liste: object) -> list[GolAraligi]:
    gelen: dict[int, dict] = {}
    if isinstance(ham_liste, list):
        for oge in ham_liste:
            if not isinstance(oge, dict):
                continue
            etiket = str(oge.get("etiket", "")).strip()
            try:
                baslangic = int(oge.get("baslangic"))
            except (TypeError, ValueError):
                baslangic = -1
            for bas, _bit, sabit_etiket in SABIT_ARALIKLAR:
                if baslangic == bas or etiket == sabit_etiket:
                    gelen[bas] = oge
                    break

    araliklar: list[GolAraligi] = []
    for baslangic, bitis, etiket in SABIT_ARALIKLAR:
        oge = gelen.get(baslangic, {})
        yorum = str(oge.get("yorum") or "").strip()
        if not yorum:
            yorum = "Model bu aralık için ayrı bir gerekçe döndürmedi."
        araliklar.append(
            GolAraligi(
                etiket=etiket,
                baslangic=baslangic,
                bitis=bitis,
                gol_olasiligi=round(_sinirla(_sayi(oge.get("gol_olasiligi")), 0, 100), 1),
                ilk_gol_olasiligi=_sinirla(_sayi(oge.get("ilk_gol_olasiligi")), 0, 100),
                yorum=yorum[:400],
            )
        )
    return araliklar


def _ilk_golu_normalize(
    araliklar: list[GolAraligi],
    golsuz: float,
) -> tuple[list[GolAraligi], float, bool]:
    """İlk gol dilimleri ile golsüz ihtimalini 100'e tamamlar."""
    degerler = [aralik.ilk_gol_olasiligi for aralik in araliklar]
    toplam = sum(degerler) + golsuz
    if toplam <= 0:
        degerler = [13.3] * 6
        golsuz = 20.2
        normalize = True
    elif 99.0 <= toplam <= 101.0:
        normalize = False
    else:
        olcek = 100.0 / toplam
        degerler = [deger * olcek for deger in degerler]
        golsuz *= olcek
        normalize = True

    yuvarlak = [round(_sinirla(deger, 0, 100), 1) for deger in degerler]
    golsuz = round(_sinirla(golsuz, 0, 100), 1)
    fark = round(100.0 - (sum(yuvarlak) + golsuz), 1)
    if fark != 0:
        golsuz = round(_sinirla(golsuz + fark, 0, 100), 1)
        fark = round(100.0 - (sum(yuvarlak) + golsuz), 1)
        if fark != 0:
            indeks = max(range(len(yuvarlak)), key=yuvarlak.__getitem__)
            yuvarlak[indeks] = round(_sinirla(yuvarlak[indeks] + fark, 0, 100), 1)

    yeni = [
        aralik.model_copy(update={"ilk_gol_olasiligi": deger})
        for aralik, deger in zip(araliklar, yuvarlak)
    ]
    return yeni, golsuz, normalize


def _json_coz(metin: str) -> dict:
    temiz = metin.strip()
    if temiz.startswith("```"):
        temiz = temiz.split("\n", 1)[-1]
        if "```" in temiz:
            temiz = temiz[: temiz.rfind("```")]
    try:
        veri = json.loads(temiz)
    except json.JSONDecodeError as exc:
        raise AnalizHatasi("Model geçerli JSON döndürmedi.") from exc
    if not isinstance(veri, dict):
        raise AnalizHatasi("Model JSON nesnesi döndürmedi.")
    return veri


def _metin_listesi(ham: object, limit: int = 5) -> list[str]:
    if not isinstance(ham, list):
        return []
    sonuc: list[str] = []
    for oge in ham:
        metin = str(oge).strip()
        if metin:
            sonuc.append(metin[:240])
        if len(sonuc) == limit:
            break
    return sonuc


def _sayi(deger: object, varsayilan: float = 0.0) -> float:
    try:
        return float(deger)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return varsayilan


def _sinirla(deger: float, alt: float, ust: float) -> float:
    return max(alt, min(ust, deger))
