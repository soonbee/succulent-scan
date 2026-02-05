<script setup lang="ts">
import UploadZone from '@/components/UploadZone.vue';

import PhotoTips from './PhotoTips.vue';
import SupportedClasses from './SupportedClasses.vue';
</script>

<template>
    <div class="space-y-6">
        <div class="mb-8 text-center">
            <h1 class="mb-3 font-serif text-3xl font-semibold text-foreground sm:text-4xl">다육식물 종류를 알려드려요</h1>
            <p class="mx-auto max-w-md text-muted-foreground">다육식물 사진을 업로드하면 다육식물이 어떤 분류인지 분석해드립니다</p>
        </div>

        <UploadZone />

        <div class="grid gap-6 md:grid-cols-2">
            <PhotoTips />
            <SupportedClasses />
        </div>
    </div>
</template>
