# Generated by Django 4.1.11 on 2023-10-26 09:31

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("Shade", "0004_alter_userprofile_pwd1"),
    ]

    operations = [
        migrations.AlterField(
            model_name="userprofile",
            name="firstname",
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name="userprofile",
            name="lastname",
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name="userprofile",
            name="pwd1",
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name="userprofile",
            name="pwd2",
            field=models.CharField(max_length=10),
        ),
    ]