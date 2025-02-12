from django.db import models


class ListAmericanCompanies(models.Model):
    id = models.IntegerField(blank=True, primary_key=True)
    ticker = models.TextField(blank=True, null=True)
    title = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'listAmericanTickers'