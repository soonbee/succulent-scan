<script setup lang="ts">
import { ImageIcon, Upload } from 'lucide-vue-next';
import { ref } from 'vue';
const isOver = ref(false);

const onFileDrop = (e: DragEvent) => {
    isOver.value = false;
    const file = e.dataTransfer?.files[0];
    // TODO
    console.log(file);
};

const onFileChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    const file = target.files?.[0];
    // TODO
    console.log(file);
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
        <input id="file-upload" type="file" class="sr-only" @change="onFileChange" />
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
</template>
