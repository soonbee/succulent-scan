<script setup lang="ts">
import { AlertCircle, ImageIcon, Upload } from 'lucide-vue-next';
import { ref } from 'vue';

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

const isOver = ref(false);
const errorMessage = ref<string | null>(null);

const processFile = (file: File | undefined) => {
    errorMessage.value = null;

    if (!file) {
        return;
    }
    if (!ACCEPTED_TYPES.includes(file.type)) {
        errorMessage.value = 'JPG, PNG, WebP 이미지 파일만 업로드할 수 있어요';
        return;
    }
    if (file.size > MAX_FILE_SIZE) {
        errorMessage.value = '10MB 이하의 파일만 업로드할 수 있어요';
        return;
    }
    console.log('file:', file);
};

const onFileDrop = (e: DragEvent) => {
    isOver.value = false;
    processFile(e.dataTransfer?.files[0]);
};

const onFileChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    processFile(target.files?.[0]);
};
</script>
<template>
    <label
        for="file-upload"
        @dragover.prevent="isOver = true"
        @dragleave="isOver = false"
        @drop.prevent="onFileDrop"
        class="relative flex min-h-70 w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition-all duration-200"
        :class="isOver ? 'scale-[1.02] border-primary bg-accent/50' : 'border-border bg-card hover:border-primary/50 hover:bg-accent/30'"
    >
        <input id="file-upload" type="file" class="sr-only" :accept="ACCEPTED_TYPES.join(',')" @change="onFileChange" />
        <div
            class="mb-4 flex h-16 w-16 items-center justify-center rounded-full transition-colors"
            :class="isOver ? 'bg-primary text-primary-foreground' : 'bg-accent text-accent-foreground'"
        >
            <Upload v-if="isOver" />
            <ImageIcon v-else />
        </div>
        <div class="text-center">
            <p class="mb-1 text-lg font-medium text-foreground">다육식물 사진을 업로드해 주세요</p>
            <p class="mb-4 text-sm text-muted-foreground">사진을 끌어다 놓거나 클릭해서 선택하세요</p>
            <p class="text-xs text-muted-foreground">JPG, PNG, WebP 파일 (최대 10MB)</p>
        </div>
    </label>
    <div v-if="errorMessage" id="upload-error" class="mt-3 flex items-center gap-2 rounded-lg bg-destructive/10 p-3 text-destructive" role="alert">
        <AlertCircle class="h-4 w-4 shrink-0" />
        <p class="text-sm">{{ errorMessage }}</p>
    </div>
</template>
