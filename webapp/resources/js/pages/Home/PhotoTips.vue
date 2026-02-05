<script setup lang="ts">
import { Focus, Scan, Square, Sun } from 'lucide-vue-next';
const tips = [
    { icon: Focus, title: '초점이 잘 맞은 사진', desc: '식물이 선명하게 보이도록 촬영해 주세요' },
    { icon: Sun, title: '밝은 환경에서 촬영', desc: '자연광이 가장 좋아요, 강한 그림자나 어두운 조명은 피해주세요' },
    { icon: Scan, title: '식물 전체가 나오게', desc: '로제트 형태나 전체 실루엣이 보이도록 찍어주세요' },
    { icon: Square, title: '단순한 배경', desc: '배경이 깔끔할수록 분류가 더 정확해요' },
];
</script>
<template>
    <div class="rounded-lg border border-border/60 bg-card text-card-foreground shadow-sm">
        <div class="flex flex-col space-y-1.5 p-6 pb-3">
            <div class="flex items-center gap-2 text-base font-medium tracking-tight">더 정확한 결과를 위한 촬영 팁</div>
        </div>
        <div class="p-6 pt-0">
            <ul class="grid grid-cols-1 gap-4">
                <li v-for="tip in tips" :key="tip.title" class="flex items-start gap-3">
                    <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent">
                        <component :is="tip.icon" class="h-4 w-4 text-accent-foreground" />
                    </div>
                    <div>
                        <p class="text-sm font-medium text-foreground">{{ tip.title }}</p>
                        <p class="text-xs text-muted-foreground">{{ tip.desc }}</p>
                    </div>
                </li>
            </ul>
        </div>
    </div>
</template>
