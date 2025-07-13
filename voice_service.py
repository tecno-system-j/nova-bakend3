from scipy.spatial.distance import pdist
from pyannote.audio import Inference
from pyannote.audio import Model
import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from typing import Dict, List, Any, Optional

class VoiceRecognitionService:
    def __init__(self):
        """Inicializar el servicio de reconocimiento de voz"""
        # Configuración para hardware limitado
        self.max_embeddings_per_speaker = 5
        self.max_speakers = 10
        self.max_audio_size_mb = 50

        hf_token = os.environ.get('hf_token')
        if not hf_token:
            raise EnvironmentError("❌ No se encontró el token 'hf_token' en las variables de entorno.")
        
        # Cargar modelo de embeddings con configuración optimizada
        try:
            print("🔄 Cargando modelo optimizado para hardware limitado...")
            self.model = Model.from_pretrained(
                "pyannote/embedding", 
                use_auth_token=hf_token
            )
            print("✅ Modelo cargado correctamente.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            raise
        
        # Base de datos dinámica
        self.speakers_db = {}
        self.db_file = "speakers_database.json"
        self.load_database()
        
        # Configuración
        self.similarity_threshold = 0.7
        self.min_samples_per_speaker = 2  # Reducido para hardware limitado
    
    def load_database(self):
        """Carga la base de datos desde archivo JSON"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.speakers_db = json.load(f)
                print(f"✅ Base de datos cargada: {len(self.speakers_db)} hablantes")
            except Exception as e:
                print(f"❌ Error cargando base de datos: {e}")
                self.speakers_db = {}
        else:
            print("📁 Creando nueva base de datos...")
            self.speakers_db = {}
    
    def save_database(self):
        """Guarda la base de datos en archivo JSON"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.speakers_db, f, indent=2)
            print("✅ Base de datos guardada")
        except Exception as e:
            print(f"❌ Error guardando base de datos: {e}")
    
    def add_speaker(self, name: str, audio_files: List[str]) -> Dict[str, Any]:
        """Agrega un nuevo hablante a la base de datos"""
        
        # Verificar límites de hardware
        if len(self.speakers_db) >= self.max_speakers:
            raise Exception(f"Límite de hablantes alcanzado ({self.max_speakers}). Elimina algunos hablantes primero.")
        
        # Verificar tamaño de archivos
        for audio_file in audio_files:
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            if file_size_mb > self.max_audio_size_mb:
                raise Exception(f"Archivo {os.path.basename(audio_file)} muy grande ({file_size_mb:.1f}MB). Máximo: {self.max_audio_size_mb}MB")
        
        if name not in self.speakers_db:
            self.speakers_db[name] = {
                'embeddings': [],
                'added_date': datetime.now().isoformat(),
                'samples_count': 0
            }
        
        print(f"🔄 Procesando {len(audio_files)} archivos para '{name}'...")
        
        embeddings_info = []
        for i, audio_file in enumerate(audio_files, 1):
            try:
                print(f"  📁 Procesando {os.path.basename(audio_file)} ({i}/{len(audio_files)})")
                embedding = self.vec(audio_file)
                
                # Validar embedding antes de procesar
                if not self._is_valid_embedding(embedding):
                    print(f"❌ Embedding inválido para {os.path.basename(audio_file)}, saltando...")
                    continue
                
                embedding_list = embedding.tolist()
                
                # Validar estadísticas antes de calcular
                try:
                    embedding_stats = {
                        'mean': float(np.mean(embedding)),
                        'std': float(np.std(embedding)),
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding))
                    }
                    
                    # Verificar que las estadísticas sean válidas
                    if any(np.isnan(v) or np.isinf(v) for v in embedding_stats.values()):
                        print(f"❌ Estadísticas inválidas para {os.path.basename(audio_file)}, saltando...")
                        continue
                        
                except Exception as e:
                    print(f"❌ Error calculando estadísticas para {os.path.basename(audio_file)}: {e}")
                    continue
                
                # Guardar información del embedding
                embedding_info = {
                    'file': os.path.basename(audio_file),
                    'embedding_shape': embedding.shape,
                    'embedding_stats': embedding_stats,
                    'embedding_preview': embedding_list[:10]  # Primeros 10 valores
                }
                embeddings_info.append(embedding_info)
                
                # Verificar límite de embeddings por hablante
                if len(self.speakers_db[name]['embeddings']) >= self.max_embeddings_per_speaker:
                    print(f"⚠️ Límite de embeddings alcanzado para '{name}' ({self.max_embeddings_per_speaker})")
                    break
                
                # Guardar en la base de datos
                self.speakers_db[name]['embeddings'].append(embedding_list)
                self.speakers_db[name]['samples_count'] += 1
                
            except Exception as e:
                print(f"❌ Error procesando {audio_file}: {e}")
        
        self.save_database()
        print(f"✅ '{name}' agregado con {len(self.speakers_db[name]['embeddings'])} muestras")
        
        return {
            'samples_added': len(self.speakers_db[name]['embeddings']),
            'embeddings_info': embeddings_info
        }
    
    def identify_speaker(self, audio_file: str) -> Dict[str, Any]:
        """Identifica al hablante con mayor precisión"""
        try:
            print(f"🔄 Analizando {os.path.basename(audio_file)}...")
            test_embedding = self.vec(audio_file)
            
            # Validar embedding de prueba
            if not self._is_valid_embedding(test_embedding):
                raise Exception("Embedding de prueba inválido o corrupto")
            
            best_match = None
            best_score = float('inf')
            confidence_scores = {}
            
            for speaker_name, speaker_data in self.speakers_db.items():
                if not speaker_data['embeddings']:
                    continue
                
                # Comparar con muestras del hablante (limitado para hardware)
                scores = []
                max_comparisons = min(3, len(speaker_data['embeddings']))  # Máximo 3 comparaciones
                
                for i, stored_embedding in enumerate(speaker_data['embeddings'][:max_comparisons]):
                    stored_embedding = np.array(stored_embedding)
                    
                    # Validar embedding almacenado
                    if not self._is_valid_embedding(stored_embedding):
                        print(f"⚠️ Embedding corrupto encontrado para {speaker_name}, saltando...")
                        continue
                    
                    try:
                        # Usar cálculo de similitud más simple para ahorrar memoria
                        distance = self._simple_cosine_distance(test_embedding, stored_embedding)
                        
                        # Validar resultado de distancia
                        if np.isnan(distance) or np.isinf(distance):
                            print(f"⚠️ Distancia inválida para {speaker_name}: {distance}")
                            continue
                            
                        scores.append(distance)
                    except Exception as e:
                        print(f"⚠️ Error calculando distancia para {speaker_name}: {e}")
                        continue
                
                if scores:  # Solo si hay scores válidos
                    min_score = min(scores)
                    confidence_scores[speaker_name] = min_score
                    
                    if min_score < best_score:
                        best_score = min_score
                        best_match = speaker_name
            
            # Verificar si hay algún match válido
            if best_match is None:
                return {
                    'speaker': 'Unknown',
                    'confidence': 0,
                    'score': float('inf'),
                    'all_scores': confidence_scores
                }
            
            # Verificar si el score es suficientemente bueno
            if best_score <= self.similarity_threshold:
                confidence = 1 - best_score
                return {
                    'speaker': best_match,
                    'confidence': confidence,
                    'score': best_score,
                    'all_scores': confidence_scores
                }
            else:
                return {
                    'speaker': 'Unknown',
                    'confidence': 0,
                    'score': best_score,
                    'all_scores': confidence_scores
                }
                
        except Exception as e:
            raise Exception(f"Error identificando hablante: {str(e)}")
    
    def _is_valid_embedding(self, embedding) -> bool:
        """Valida si un embedding es válido"""
        try:
            # Verificar que no sea None
            if embedding is None:
                return False
            
            # Convertir a numpy array si es necesario
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Verificar que no esté vacío
            if embedding.size == 0:
                return False
            
            # Verificar que no tenga valores NaN
            if np.any(np.isnan(embedding)):
                return False
            
            # Verificar que no tenga valores infinitos
            if np.any(np.isinf(embedding)):
                return False
            
            # Verificar que no sea todo ceros
            if np.all(embedding == 0):
                return False
            
            return True
        except Exception:
            return False
    
    def _simple_cosine_distance(self, vec1, vec2):
        """Cálculo optimizado de distancia coseno para hardware limitado"""
        try:
            # Normalizar vectores
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
            
            # Calcular similitud coseno
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Convertir a distancia (1 - similitud)
            distance = 1 - similarity
            
            # Asegurar que esté en rango [0, 1]
            distance = max(0, min(1, distance))
            
            return distance
        except Exception as e:
            print(f"⚠️ Error en cálculo optimizado: {e}")
            return 1.0  # Máxima distancia en caso de error
    
    def get_speaker_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la base de datos"""
        stats = {
            'total_speakers': len(self.speakers_db),
            'total_samples': sum(s['samples_count'] for s in self.speakers_db.values()),
            'speakers': {}
        }
        
        for name, data in self.speakers_db.items():
            stats['speakers'][name] = {
                'name': name,
                'samples_count': data['samples_count'],
                'added_date': data['added_date']
            }
        
        return stats
    
    def get_speakers_list(self) -> List[str]:
        """Obtiene lista de nombres de hablantes"""
        return list(self.speakers_db.keys())
    
    def delete_speaker(self, speaker_name: str) -> Dict[str, Any]:
        """Elimina un hablante de la base de datos"""
        if speaker_name not in self.speakers_db:
            raise Exception(f"Hablante '{speaker_name}' no encontrado")
        
        samples_removed = self.speakers_db[speaker_name]['samples_count']
        del self.speakers_db[speaker_name]
        self.save_database()
        
        return {
            'samples_removed': samples_removed
        }
    
    def extract_embedding(self, audio_file: str) -> Dict[str, Any]:
        """Extrae embedding de un archivo de audio sin guardarlo"""
        try:
            print(f"🔄 Extrayendo embedding de {os.path.basename(audio_file)}...")
            embedding = self.vec(audio_file)
            
            embedding_info = {
                'file': os.path.basename(audio_file),
                'embedding_shape': embedding.shape,
                'embedding_stats': {
                    'mean': float(np.mean(embedding)),
                    'std': float(np.std(embedding)),
                    'min': float(np.min(embedding)),
                    'max': float(np.max(embedding))
                },
                'embedding_preview': embedding.tolist()[:10]
            }
            
            return embedding_info
        except Exception as e:
            raise Exception(f"Error extrayendo embedding: {str(e)}")
    
    def plot_vectors(self, save_path: Optional[str] = None) -> str:
        """Crea gráficos de los vectores almacenados"""
        if not self.speakers_db:
            raise Exception("No hay hablantes en la base de datos para graficar")
        
        print("🔄 Generando gráficos de vectores...")
        
        # Preparar datos
        all_embeddings = []
        speaker_names = []
        
        for speaker_name, speaker_data in self.speakers_db.items():
            for embedding in speaker_data['embeddings']:
                all_embeddings.append(embedding)
                speaker_names.append(speaker_name)
        
        if not all_embeddings:
            raise Exception("No hay embeddings para graficar")
        
        all_embeddings = np.array(all_embeddings)
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Análisis de Embedding Vectors', fontsize=16)
        
        # 1. Distribución de valores
        axes[0, 0].hist(all_embeddings.flatten(), bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribución de Valores de Embeddings')
        axes[0, 0].set_xlabel('Valor')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # 2. Heatmap de embeddings por hablante
        for speaker_name in self.speakers_db.keys():
            speaker_embeddings = []
            for embedding in self.speakers_db[speaker_name]['embeddings']:
                speaker_embeddings.append(embedding)
            
            if speaker_embeddings:
                speaker_embeddings = np.array(speaker_embeddings)
                axes[0, 1].imshow(speaker_embeddings, aspect='auto', cmap='viridis', alpha=0.8)
                axes[0, 1].set_title('Heatmap de Embeddings')
                axes[0, 1].set_xlabel('Dimensión del Vector')
                axes[0, 1].set_ylabel('Muestras')
        
        # 3. Estadísticas por hablante
        speaker_stats = []
        speaker_names_list = []
        for speaker_name, speaker_data in self.speakers_db.items():
            if speaker_data['embeddings']:
                embeddings = np.array(speaker_data['embeddings'])
                speaker_stats.append([
                    np.mean(embeddings),
                    np.std(embeddings),
                    np.min(embeddings),
                    np.max(embeddings)
                ])
                speaker_names_list.append(speaker_name)
        
        if speaker_stats:
            speaker_stats = np.array(speaker_stats)
            x_pos = np.arange(len(speaker_names_list))
            width = 0.2
            
            axes[1, 0].bar(x_pos - width, speaker_stats[:, 0], width, label='Media', alpha=0.8)
            axes[1, 0].bar(x_pos, speaker_stats[:, 1], width, label='Desv. Est.', alpha=0.8)
            axes[1, 0].bar(x_pos + width, speaker_stats[:, 2], width, label='Mínimo', alpha=0.8)
            axes[1, 0].bar(x_pos + 2*width, speaker_stats[:, 3], width, label='Máximo', alpha=0.8)
            
            axes[1, 0].set_title('Estadísticas por Hablante')
            axes[1, 0].set_ylabel('Valor')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(speaker_names_list, rotation=45)
            axes[1, 0].legend()
        
        # 4. PCA para visualización 2D
        if len(all_embeddings) > 1:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(all_embeddings)
            
            for speaker_name in self.speakers_db.keys():
                speaker_indices = [j for j, name in enumerate(speaker_names) if name == speaker_name]
                if speaker_indices:
                    axes[1, 1].scatter(
                        embeddings_2d[speaker_indices, 0], 
                        embeddings_2d[speaker_indices, 1], 
                        label=speaker_name, 
                        alpha=0.7,
                        s=50
                    )
            
            axes[1, 1].set_title('Visualización PCA (2D)')
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC2')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Guardar gráfico
        if save_path is None:
            save_path = f"vectors_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado como: {save_path}")
        
        return save_path 
