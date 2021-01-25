# Generated by Django 3.1.5 on 2021-01-23 11:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Leaf',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_user', models.IntegerField()),
                ('name', models.CharField(default='', max_length=50)),
                ('leaf_image_url', models.ImageField(upload_to='images/')),
                ('analyze', models.BooleanField(default=False)),
                ('prediction', models.IntegerField()),
                ('url', models.CharField(default='', max_length=100)),
            ],
        ),
    ]