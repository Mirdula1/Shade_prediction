# Generated by Django 4.1.11 on 2023-10-17 15:18

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="DyeQualityPrediction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("Lf", models.FloatField()),
                ("Af", models.FloatField()),
                ("Bf", models.FloatField()),
                ("pH", models.FloatField()),
                ("Temp", models.IntegerField()),
                ("Substrate", models.TextField()),
                ("Thread", models.TextField()),
                ("Thickness", models.FloatField()),
                ("thread_group", models.TextField()),
                ("D_Duration", models.IntegerField()),
                ("Fastness_Type", models.TextField()),
                ("Washings", models.IntegerField()),
                ("Chemical", models.TextField()),
                ("Chemical_Conc", models.FloatField()),
                ("Lubricant", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="ShadePrediction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("li", models.FloatField()),
                ("ai", models.FloatField()),
                ("bi", models.FloatField()),
                ("Concentration", models.FloatField()),
                ("pH", models.FloatField()),
                ("Temp", models.IntegerField()),
                ("WaterBathRatio", models.FloatField()),
                ("DyeingMethod", models.TextField()),
                ("Duration", models.IntegerField()),
                ("Thread", models.TextField()),
                ("Thickness", models.FloatField()),
                ("thread_group", models.TextField()),
                ("Abs_coeff", models.FloatField()),
                ("Lf", models.FloatField()),
                ("Af", models.FloatField()),
                ("Bf", models.FloatField()),
                ("delta_e", models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name="UserProfile",
            fields=[
                (
                    "username",
                    models.CharField(max_length=50, primary_key=True, serialize=False),
                ),
                ("firstname", models.CharField(max_length=50)),
                ("lastname", models.CharField(max_length=50)),
                ("address", models.CharField(max_length=100)),
                ("designation", models.CharField(max_length=10)),
                ("phone", models.CharField(max_length=10)),
                ("email", models.CharField(max_length=20)),
                ("pwd1", models.CharField(max_length=20)),
                ("pwd2", models.CharField(max_length=20)),
                ("DOB", models.DateField()),
            ],
        ),
    ]
