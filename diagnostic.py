#!/usr/bin/env python3
"""
Script de diagnóstico para el sistema de reconocimiento de voz
"""

import os
import sys
import numpy as np
from pyannote.audio import Model, Inference
import json

def check_audio_file(file_path):
    """Verificar si un archivo de audio es válido"""
    try:
        if not os.path.exists(file_path):
            return False, "Archivo no existe"
        
        # Verificar tamaño
        size = os.path.getsize(file_path)
        if size == 0:
            return False, "Archivo vacío"
        
        # Verificar extensión
        valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in valid_extensions:
            return False, f"Extensión no soportada: {ext}"
        
        return True, f"Archivo válido ({size} bytes)"
    except Exception as e:
        return False, f"Error verificando archivo: {e}"

def test_model_loading():
    """Probar carga del modelo"""
    try:
        print("🔄 Cargando modelo de embeddings...")
        model = Model.from_pretrained(
            "pyannote/embedding", 
            use_auth_token="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"
        )
        vec = Inference(model, window='whole')
        print("✅ Modelo cargado correctamente")
        return True, vec
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False, None

def test_embedding_extraction(vec, test_file):
    """Probar extracción de embedding"""
    try:
        print(f"🔄 Extrayendo embedding de {test_file}...")
        embedding = vec(test_file)
        
        # Validar embedding
        if embedding is None:
            return False, "Embedding es None"
        
        if not isinstance(embedding, np.ndarray):
            return False, f"Embedding no es numpy array: {type(embedding)}"
        
        if embedding.size == 0:
            return False, "Embedding vacío"
        
        if np.any(np.isnan(embedding)):
            return False, "Embedding contiene valores NaN"
        
        if np.any(np.isinf(embedding)):
            return False, "Embedding contiene valores infinitos"
        
        if np.all(embedding == 0):
            return False, "Embedding es todo ceros"
        
        print(f"✅ Embedding válido: forma {embedding.shape}")
        print(f"   Estadísticas: mean={np.mean(embedding):.6f}, std={np.std(embedding):.6f}")
        return True, embedding
        
    except Exception as e:
        return False, f"Error extrayendo embedding: {e}"

def test_database_integrity():
    """Verificar integridad de la base de datos"""
    db_file = "speakers_database.json"
    
    if not os.path.exists(db_file):
        print("📭 Base de datos no existe")
        return True
    
    try:
        with open(db_file, 'r') as f:
            db = json.load(f)
        
        print(f"📊 Base de datos: {len(db)} hablantes")
        
        total_embeddings = 0
        corrupted_embeddings = 0
        
        for speaker_name, speaker_data in db.items():
            embeddings = speaker_data.get('embeddings', [])
            total_embeddings += len(embeddings)
            
            for i, embedding in enumerate(embeddings):
                try:
                    embedding_array = np.array(embedding)
                    
                    # Verificar embedding
                    if (embedding_array.size == 0 or 
                        np.any(np.isnan(embedding_array)) or 
                        np.any(np.isinf(embedding_array)) or
                        np.all(embedding_array == 0)):
                        corrupted_embeddings += 1
                        print(f"⚠️ Embedding corrupto: {speaker_name}[{i}]")
                        
                except Exception as e:
                    corrupted_embeddings += 1
                    print(f"⚠️ Error procesando embedding: {speaker_name}[{i}]: {e}")
        
        print(f"📈 Total embeddings: {total_embeddings}")
        print(f"❌ Embeddings corruptos: {corrupted_embeddings}")
        
        if corrupted_embeddings > 0:
            print("💡 Considera regenerar la base de datos")
            return False
        else:
            print("✅ Base de datos íntegra")
            return True
            
    except Exception as e:
        print(f"❌ Error verificando base de datos: {e}")
        return False

def test_similarity_calculation(vec, test_file):
    """Probar cálculo de similitud"""
    try:
        from scipy.spatial.distance import pdist
        
        print("🔄 Probando cálculo de similitud...")
        
        # Extraer dos embeddings del mismo archivo
        embedding1 = vec(test_file)
        embedding2 = vec(test_file)
        
        # Calcular distancia coseno
        distance = pdist([embedding1, embedding2], metric='cosine')[0]
        
        print(f"✅ Distancia coseno calculada: {distance:.6f}")
        
        if np.isnan(distance) or np.isinf(distance):
            return False, f"Distancia inválida: {distance}"
        
        return True, distance
        
    except Exception as e:
        return False, f"Error calculando similitud: {e}"

def main():
    """Función principal de diagnóstico"""
    print("🔍 Diagnóstico del Sistema de Reconocimiento de Voz")
    print("="*60)
    
    # 1. Verificar archivos de prueba
    test_files = [
        "test_audio.wav",
        "sample1.wav", 
        "sample2.wav"
    ]
    
    print("\n📁 Verificando archivos de prueba...")
    valid_files = []
    for test_file in test_files:
        is_valid, message = check_audio_file(test_file)
        status = "✅" if is_valid else "❌"
        print(f"  {status} {test_file}: {message}")
        if is_valid:
            valid_files.append(test_file)
    
    if not valid_files:
        print("⚠️ No hay archivos de prueba válidos")
        print("💡 Coloca algunos archivos de audio en el directorio")
        return
    
    # 2. Probar carga del modelo
    print("\n🤖 Probando carga del modelo...")
    model_ok, vec = test_model_loading()
    if not model_ok:
        print("❌ No se puede continuar sin el modelo")
        return
    
    # 3. Probar extracción de embeddings
    print(f"\n🎵 Probando extracción de embeddings...")
    test_file = valid_files[0]
    embedding_ok, embedding = test_embedding_extraction(vec, test_file)
    if not embedding_ok:
        print(f"❌ Error con embedding: {embedding}")
        return
    
    # 4. Probar cálculo de similitud
    similarity_ok, distance = test_similarity_calculation(vec, test_file)
    if not similarity_ok:
        print(f"❌ Error con similitud: {distance}")
        return
    
    # 5. Verificar base de datos
    print("\n💾 Verificando base de datos...")
    db_ok = test_database_integrity()
    
    # Resumen
    print("\n" + "="*60)
    print("📋 RESUMEN DEL DIAGNÓSTICO")
    print("="*60)
    
    print(f"📁 Archivos válidos: {len(valid_files)}/{len(test_files)}")
    print(f"🤖 Modelo: {'✅' if model_ok else '❌'}")
    print(f"🎵 Embeddings: {'✅' if embedding_ok else '❌'}")
    print(f"📊 Similitud: {'✅' if similarity_ok else '❌'}")
    print(f"💾 Base de datos: {'✅' if db_ok else '❌'}")
    
    if all([model_ok, embedding_ok, similarity_ok, db_ok]):
        print("\n🎉 Sistema funcionando correctamente")
    else:
        print("\n⚠️ Se encontraron problemas. Revisa los errores arriba.")

if __name__ == "__main__":
    main() 