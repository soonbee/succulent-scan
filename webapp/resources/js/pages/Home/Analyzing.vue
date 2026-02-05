<script setup lang="ts">
import { Leaf } from 'lucide-vue-next';
import { onMounted, onUnmounted, ref } from 'vue';

const messages = [
    '잎 모양을 살펴보고 있어요...',
    '로제트 형태를 분석하는 중이에요...',
    '생장 특징을 평가하고 있어요...',
    '비슷한 종류와 비교하고 있어요...',
];
const count = ref(0);
let timerId: number | undefined;

onMounted(() => {
    timerId = window.setInterval(() => {
        count.value = (count.value + 1) % messages.length;
    }, 1500);
});

onUnmounted(() => {
    if (timerId) {
        clearInterval(timerId);
    }
});
</script>
<template>
    <div class="flex flex-col items-center justify-center px-4 py-16">
        <div class="relative">
            <div class="flex h-20 w-20 items-center justify-center rounded-full bg-accent">
                <Leaf class="h-10 w-10 animate-pulse text-primary" />
            </div>
            <div class="absolute inset-0 h-20 w-20 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
        </div>
        <h2 class="mt-8 font-serif text-xl font-medium text-foreground">사진 분석중</h2>
        <p class="mt-2 min-h-6 text-center text-muted-foreground transition-opacity">
            {{ messages[count] }}
        </p>
        <div class="mt-6 flex gap-1.5">
            <div class="h-2 w-2 animate-bounce rounded-full bg-primary" style="animation-delay: 0ms"></div>
            <div class="h-2 w-2 animate-bounce rounded-full bg-primary" style="animation-delay: 150ms"></div>
            <div class="h-2 w-2 animate-bounce rounded-full bg-primary" style="animation-delay: 300ms"></div>
        </div>
    </div>
</template>
