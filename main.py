from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import tempfile
import shutil

from voice_service import VoiceRecognitionService
from models import (
    SpeakerAddRequest, 
    SpeakerAddResponse, 
    SpeakerIdentifyRequest, 
    SpeakerIdentifyResponse,
    SpeakerStats,
    ThresholdConfig
)

app = FastAPI(
    title="Voice Recognition API",
    description="API para reconocimiento de voz usando embeddings",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicio de voz
voice_service = VoiceRecognitionService()

@app.on_event("startup")
async def startup_event():
    """Evento de inicio del servidor"""
    try:
        print("🎤 Servidor de Reconocimiento de Voz iniciado")
        print("📊 Base de datos cargada")
        
        # Verificar que el modelo se cargue correctamente
        test_embedding = voice_service.vec("test_audio.wav") if os.path.exists("test_audio.wav") else None
        if test_embedding is not None:
            print("✅ Modelo de embeddings cargado correctamente")
        else:
            print("⚠️ Modelo cargado (sin archivo de prueba)")
            
    except Exception as e:
        print(f"❌ Error inicializando servicio: {e}")
        print("💡 Verifica que el token de Hugging Face sea válido")

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Voice Recognition API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Verificar estado del servidor"""
    return {"status": "healthy", "service": "voice_recognition"}

@app.post("/speakers/add", response_model=SpeakerAddResponse)
async def add_speaker(request: SpeakerAddRequest):
    """Agregar un nuevo hablante a la base de datos"""
    try:
        result = voice_service.add_speaker(
            name=request.name,
            audio_files=request.audio_files
        )
        return SpeakerAddResponse(
            success=True,
            speaker_name=request.name,
            samples_added=result['samples_added'],
            embeddings_info=result['embeddings_info'],
            message=f"Hablante '{request.name}' agregado exitosamente"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/speakers/identify", response_model=SpeakerIdentifyResponse)
async def identify_speaker(request: SpeakerIdentifyRequest):
    """Identificar un hablante en un archivo de audio"""
    try:
        result = voice_service.identify_speaker(request.audio_file)
        return SpeakerIdentifyResponse(
            success=True,
            speaker=result['speaker'],
            confidence=result['confidence'],
            score=result['score'],
            all_scores=result['all_scores'],
            message=f"Hablante identificado: {result['speaker']}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Subir archivo de audio"""
    try:
        # Crear directorio temporal si no existe
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Guardar archivo
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "success": True,
            "file_path": file_path,
            "filename": file.filename,
            "size": os.path.getsize(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error subiendo archivo: {str(e)}")

@app.get("/stats", response_model=SpeakerStats)
async def get_stats():
    """Obtener estadísticas de la base de datos"""
    try:
        stats = voice_service.get_speaker_stats()
        return SpeakerStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/threshold")
async def get_threshold():
    """Obtener umbral actual de similitud"""
    return {
        "threshold": voice_service.similarity_threshold,
        "min_samples": voice_service.min_samples_per_speaker
    }

@app.put("/threshold")
async def update_threshold(config: ThresholdConfig):
    """Actualizar umbral de similitud"""
    try:
        voice_service.similarity_threshold = config.threshold
        voice_service.min_samples_per_speaker = config.min_samples
        return {
            "success": True,
            "threshold": config.threshold,
            "min_samples": config.min_samples,
            "message": "Umbral actualizado exitosamente"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/speakers")
async def get_speakers():
    """Obtener lista de hablantes en la base de datos"""
    try:
        speakers = voice_service.get_speakers_list()
        return {
            "success": True,
            "speakers": speakers,
            "total": len(speakers)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/speakers/{speaker_name}")
async def delete_speaker(speaker_name: str):
    """Eliminar un hablante de la base de datos"""
    try:
        result = voice_service.delete_speaker(speaker_name)
        return {
            "success": True,
            "message": f"Hablante '{speaker_name}' eliminado exitosamente",
            "samples_removed": result['samples_removed']
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/extract/embedding")
async def extract_embedding(request: SpeakerIdentifyRequest):
    """Extraer embedding de un archivo de audio sin guardarlo"""
    try:
        embedding_info = voice_service.extract_embedding(request.audio_file)
        return {
            "success": True,
            "embedding_info": embedding_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Configurar límites de memoria para hardware limitado
    import resource
    try:
        # Limitar memoria a 2GB para AMD Sempron
        memory_limit_gb = 2
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_gb * 1024 * 1024 * 1024, -1))
        print(f"✅ Límite de memoria configurado: {memory_limit_gb}GB")
    except Exception as e:
        print(f"⚠️ No se pudo configurar límite de memoria: {e}")
    
    # Configuración optimizada para hardware limitado
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Desactivar reload para ahorrar memoria
        log_level="warning",  # Reducir logging
        access_log=False,  # Desactivar access log
        workers=1,  # Un solo worker
        loop="asyncio"
    ) 