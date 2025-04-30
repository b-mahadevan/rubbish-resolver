from django import forms

class WasteImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload an image of waste',
        help_text='Upload an image to classify waste type'
    )