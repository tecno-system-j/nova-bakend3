#!/usr/bin/env python3
"""
Configuración optimizada para hardware limitado (AMD Sempron + 4GB RAM)
"""

import os
import psutil
import gc
import numpy as np

class HardwareOptimizer:
    def __init__(self):
        self.memory_limit_gb = 2
        self.cpu_limit_percent = 80
        self.disk_space_limit_gb = 1
        
    def check_system_resources(self):
        """Verificar recursos del sistema"""
        print("🔍 Verificando recursos del sistema...")
        
        # Memoria RAM
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        print(f"💾 RAM Total: {memory_gb:.1f}GB")
        print(f"💾 RAM Disponible: {memory_available_gb:.1f}GB")
        print(f"💾 RAM Usada: {memory.percent:.1f}%")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"🖥️ CPU Uso: {cpu_percent:.1f}%")
        
        # Disco
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        print(f"💿 Disco Libre: {disk_free_gb:.1f}GB")
        
        # Verificar si hay suficientes recursos
        warnings = []
        
        if memory_available_gb < self.memory_limit_gb:
            warnings.append(f"⚠️ RAM insuficiente: {memory_available_gb:.1f}GB < {self.memory_limit_gb}GB")
        
        if cpu_percent > self.cpu_limit_percent:
            warnings.append(f"⚠️ CPU muy ocupada: {cpu_percent:.1f}% > {self.cpu_limit_percent}%")
        
        if disk_free_gb < self.disk_space_limit_gb:
            warnings.append(f"⚠️ Poco espacio en disco: {disk_free_gb:.1f}GB < {self.disk_space_limit_gb}GB")
        
        if warnings:
            print("\n⚠️ ADVERTENCIAS:")
            for warning in warnings:
                print(f"  {warning}")
            return False
        else:
            print("✅ Recursos del sistema OK")
            return True
    
    def optimize_memory(self):
        """Optimizar uso de memoria"""
        print("🔄 Optimizando memoria...")
        
        # Forzar garbage collection
        gc.collect()
        
        # Limpiar caché de numpy
        if hasattr(np, 'clear_cache'):
            np.clear_cache()
        
        # Verificar memoria después de optimización
        memory = psutil.virtual_memory()
        memory_available_gb = memory.available / (1024**3)
        print(f"✅ Memoria disponible después de optimización: {memory_available_gb:.1f}GB")
    
    def get_optimal_settings(self):
        """Obtener configuración óptima para el hardware"""
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb <= 2:
            # Configuración para 2GB RAM
            return {
                'max_speakers': 5,
                'max_embeddings_per_speaker': 3,
                'max_audio_size_mb': 25,
                'max_comparisons': 2,
                'similarity_threshold': 0.8,  # Más estricto para menos comparaciones
                'min_samples_per_speaker': 1
            }
        elif memory_gb <= 4:
            # Configuración para 4GB RAM
            return {
                'max_speakers': 10,
                'max_embeddings_per_speaker': 5,
                'max_audio_size_mb': 50,
                'max_comparisons': 3,
                'similarity_threshold': 0.7,
                'min_samples_per_speaker': 2
            }
        else:
            # Configuración para 8GB+ RAM
            return {
                'max_speakers': 20,
                'max_embeddings_per_speaker': 10,
                'max_audio_size_mb': 100,
                'max_comparisons': 5,
                'similarity_threshold': 0.7,
                'min_samples_per_speaker': 3
            }
    
    def apply_optimizations(self):
        """Aplicar todas las optimizaciones"""
        print("⚙️ Aplicando optimizaciones para hardware limitado...")
        
        # Verificar recursos
        if not self.check_system_resources():
            print("⚠️ Sistema con recursos limitados detectado")
        
        # Optimizar memoria
        self.optimize_memory()
        
        # Obtener configuración óptima
        settings = self.get_optimal_settings()
        
        print("\n📋 Configuración óptima:")
        for key, value in settings.items():
            print(f"  • {key}: {value}")
        
        return settings

def main():
    """Función principal"""
    print("🔧 Optimizador para Hardware Limitado")
    print("="*50)
    
    optimizer = HardwareOptimizer()
    settings = optimizer.apply_optimizations()
    
    print("\n✅ Optimizaciones aplicadas")
    print("💡 Usa esta configuración en voice_service.py")

if __name__ == "__main__":
    main() 