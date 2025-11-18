from django import forms
from .models import AudioFile
from django import forms
from .models import Pasien

class AudioForm(forms.ModelForm):
    class Meta:
        model = AudioFile
        fields = ['judul', 'file']

class PasienForm(forms.ModelForm):
    class Meta:
        model = Pasien
        fields = '__all__'