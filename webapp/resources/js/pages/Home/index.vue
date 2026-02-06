<script setup lang="ts">
import { router } from '@inertiajs/vue3';
import { ref } from 'vue';

import UploadZone from '@/components/UploadZone.vue';
import { useUploadedImage } from '@/composables/useUploadedImage';
import { store } from '@/routes/analyze';

import Analyzing from './Analyzing.vue';
import PhotoTips from './PhotoTips.vue';
import SupportedClasses from './SupportedClasses.vue';

const isLoading = ref(false);
const { setImage } = useUploadedImage();

const onFileSelect = async (file: File) => {
    isLoading.value = true;
    setImage(file);

    router.post(store.url(), { image: file }, {
        forceFormData: true,
        onFinish: () => {
            isLoading.value = false;
        },
    });
};
</script>

<template>
    <div v-if="isLoading">
        <Analyzing />
    </div>
    <div v-else class="space-y-6">
        <div class="mb-8 text-center">
            <h1 class="mb-3 font-serif text-3xl font-semibold text-foreground sm:text-4xl">다육식물 종류를 알려드려요</h1>
            <p class="mx-auto max-w-md text-muted-foreground">다육식물 사진을 업로드하면 다육식물이 어떤 분류인지 분석해드립니다</p>
        </div>

        <UploadZone @select="onFileSelect" />

        <div class="grid gap-6 md:grid-cols-2">
            <PhotoTips />
            <SupportedClasses />
        </div>
    </div>
</template>
