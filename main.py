"""İddaa AI Analysis — FastAPI sunucusu.

Yerel çalıştırma (backend klasöründen):
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Render: kök klasör backend, başlatma komutu Procfile içindedir.
Ortam değişkeni GEMINI_API_KEY tanımlı olmalıdır.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from analiz_engine import MODEL_ADI, AnalizHatasi, AnalizMotoru, MacAnalizi
from bulten import Bulten, BultenHatasi, mac_bul, bulteni_getir
from mac_verisi import derin_baglam

load_dotenv(Path(__file__).resolve().parent / ".env")


def gemini_anahtari() -> str | None:
    for ad in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        deger = (os.getenv(ad) or "").strip()
        if deger:
            return deger
    return None


app = FastAPI(
    title="İddaa AI Analysis",
    description=(
        "Güncel maç bülteni ve seçilen maçın 1X2, alt/üst, korner ve dakika "
        "analizini üreten servis. Çıktılar bahis tavsiyesi değildir."
    ),
    version="0.2.0",
)

# Vercel, Render ve yerel istemcilerin tamamı bu API'ye erişebilir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

motor = AnalizMotoru(api_key=gemini_anahtari())


class MacIstegi(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
{"mac_id": "tur.1-12345"}
            ]
        }
    )

    mac_id: Optional[str] = Field(default=None, max_length=80)
    ev_sahibi: str = Field(default="", max_length=80)
    deplasman: str = Field(default="", max_length=80)
    lig: str = Field(default="Belirtilmedi", max_length=80)
    tarih: Optional[str] = Field(default=None, max_length=40)
    notlar: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("ev_sahibi", "deplasman", mode="before")
    @classmethod
    def takimi_temizle(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value

    @field_validator("lig", mode="before")
    @classmethod
    def ligi_temizle(cls, value: object) -> str:
        if not isinstance(value, str):
            return "Belirtilmedi"
        value = " ".join(value.split())
        return value or "Belirtilmedi"

    @field_validator("tarih", "notlar", mode="before")
    @classmethod
    def bosu_yok_say(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip()
        return value or None


@app.get("/")
def kok() -> dict[str, str]:
    return {
        "servis": "İddaa AI Analysis",
        "durum": "hazir",
        "dokuman": "/docs",
        "bulten": "/api/v1/bulten",
        "analiz": "/api/v1/analiz",
    }


@app.get("/api/v1/saglik")
def saglik() -> dict[str, object]:
    return {
        "durum": "saglikli",
        "model": MODEL_ADI,
        "api_anahtari_tanimli": motor.anahtar_var,
    }


@app.get("/api/v1/bulten", response_model=Bulten)
def bulten() -> Bulten:
    try:
        return bulteni_getir()
    except BultenHatasi as exc:
        raise HTTPException(status_code=exc.durum_kodu, detail=str(exc)) from exc


@app.post("/api/v1/analiz", response_model=MacAnalizi)
def analiz(istek: MacIstegi) -> MacAnalizi:
    ev_sahibi = istek.ev_sahibi
    deplasman = istek.deplasman
    lig = istek.lig
    tarih = istek.tarih
    if istek.mac_id:
        mac = mac_bul(istek.mac_id)
        if mac is None:
            raise HTTPException(status_code=404, detail="Bu maç güncel bültende yok.")
        ev_sahibi = mac.ev_sahibi
        deplasman = mac.deplasman
        lig = mac.lig
        tarih = mac.baslangic if mac.durum == "yaklasan" else None
        baglam = derin_baglam(mac)
        notlar = baglam.gemini_metni
        if istek.notlar:
            notlar = f"{notlar}\nEk not: {istek.notlar}"
    else:
        baglam = None
        notlar = istek.notlar
    if len(ev_sahibi) < 2 or len(deplasman) < 2:
        raise HTTPException(status_code=400, detail="Ev sahibi ve deplasman gerekli.")
    if ev_sahibi.casefold() == deplasman.casefold():
        raise HTTPException(status_code=400, detail="Ev sahibi ve deplasman aynı takım olamaz.")
    try:
        return motor.analiz_et(
            ev_sahibi=ev_sahibi,
            deplasman=deplasman,
            lig=lig,
            tarih=tarih,
            notlar=notlar,
            baglam=baglam,
        )
    except AnalizHatasi as exc:
        raise HTTPException(status_code=exc.durum_kodu, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
