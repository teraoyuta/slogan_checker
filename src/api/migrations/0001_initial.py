# Generated by Django 2.2.12 on 2024-04-24 09:28

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='slogans',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('slogan_sentence', models.CharField(max_length=200)),
                ('slogan_kana', models.CharField(max_length=200)),
                ('vector', models.TextField()),
            ],
            options={
                'db_table': 'slogans',
            },
        ),
    ]
