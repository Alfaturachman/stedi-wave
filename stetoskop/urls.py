"""
URL configuration for stetoskop project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from audio import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='beranda'),
    path('tambah-pasien/', views.tambah_pasien, name='tambah_pasien'),
    path('periksa-pasien/', views.upload_audio, name='upload_audio'),
    path('daftar-pasien/', views.daftar_pasien, name='daftar_pasien'),
    path('riwayat-pasien/<str:patient_id>/', views.riwayat_pasien, name='riwayat_pasien'),
    path('riwayat-pasien/delete/<str:exam_id>/', views.delete_riwayat, name='delete_riwayat'),
    path('analyze_lung_sound_enhanced/', views.analyze_lung_sound_enhanced, name='analyze_lung_sound_enhanced'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)