# Generated by Django 4.1.11 on 2023-10-26 09:27

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("Shade", "0003_alter_userprofile_email_alter_userprofile_phone_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="userprofile",
            name="pwd1",
            field=models.CharField(max_length=20),
        ),
    ]
