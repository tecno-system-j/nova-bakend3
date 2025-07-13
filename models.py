from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class SpeakerAddRequest(BaseModel):
    """Request para agregar un nuevo hablante"""
    name: str = Field(..., description="Nombre del hablante")
    audio_files: List[str] = Field(..., description="Lista de rutas de archivos de audio")

class EmbeddingInfo(BaseModel):
    """Información de un embedding extraído"""
    file: str
    embedding_shape: List[int]
    embedding_stats: Dict[str, float]
    embedding_preview: List[float] = Field(..., description="Primeros 10 valores del embedding")

class SpeakerAddResponse(BaseModel):
    """Response para agregar hablante"""
    success: bool
    speaker_name: str
    samples_added: int
    embeddings_info: List[EmbeddingInfo]
    message: str

class SpeakerIdentifyRequest(BaseModel):
    """Request para identificar un hablante"""
    audio_file: str = Field(..., description="Ruta del archivo de audio a identificar")

class SpeakerIdentifyResponse(BaseModel):
    """Response para identificación de hablante"""
    success: bool
    speaker: str
    confidence: float
    score: float
    all_scores: Dict[str, float]
    message: str

class SpeakerInfo(BaseModel):
    """Información de un hablante"""
    name: str
    samples_count: int
    added_date: str

class SpeakerStats(BaseModel):
    """Estadísticas de la base de datos"""
    total_speakers: int
    total_samples: int
    speakers: Dict[str, SpeakerInfo]

class ThresholdConfig(BaseModel):
    """Configuración del umbral de similitud"""
    threshold: float = Field(..., ge=0.1, le=1.0, description="Umbral de similitud (0.1-1.0)")
    min_samples: int = Field(..., ge=1, description="Mínimo de muestras por hablante")

class HealthResponse(BaseModel):
    """Response de health check"""
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.now)

class UploadResponse(BaseModel):
    """Response para subida de archivos"""
    success: bool
    file_path: str
    filename: str
    size: int

class DeleteSpeakerResponse(BaseModel):
    """Response para eliminar hablante"""
    success: bool
    message: str
    samples_removed: int

class ExtractEmbeddingResponse(BaseModel):
    """Response para extracción de embedding"""
    success: bool
    embedding_info: EmbeddingInfo

class SpeakersListResponse(BaseModel):
    """Response para lista de hablantes"""
    success: bool
    speakers: List[str]
    total: int 