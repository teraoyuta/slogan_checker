from django.db import models

# Create your models here.

class slogans(models.Model):
    slogan_sentence = models.CharField(max_length=200)
    slogan_kana = models.CharField(max_length=200)
    vector = models.TextField()

    class Meta:
        db_table = 'slogans'
