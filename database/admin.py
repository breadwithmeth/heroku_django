from django.contrib import admin
from .models import citizens, upload_files, search_upload
# Register your models here.
admin.site.register(citizens)

admin.site.register(upload_files)

admin.site.register(search_upload)