<script setup lang="ts">
import { Link } from '@inertiajs/vue3';
import { ChevronRight, RefreshCcw, TriangleAlert } from 'lucide-vue-next';

import { useUploadedImage } from '@/composables/useUploadedImage';
import { home } from '@/routes';

interface ResultItem {
    ko: string;
    en: string;
    acc: number;
}

defineProps<{
    reliable: boolean;
    results: ResultItem[];
}>();

const { imageUrl } = useUploadedImage();
</script>
<template>
    <div v-if="!reliable">
        <p>[WIP] UNKNWON RESULT</p>
    </div>
    <div v-else class="mx-auto w-full max-w-2xl">
        <div class="mb-8 text-center">
            <h2 class="mb-2 font-serif text-2xl font-semibold text-foreground">다육이 분류 결과</h2>
            <p class="text-muted-foreground">
                가장 가능성이 높은 종류는 <span class="font-medium text-primary">{{ results[0].ko }}({{ results[0].en }})</span> 이에요
            </p>
        </div>
        <div class="mb-6 overflow-hidden rounded-lg border bg-card text-card-foreground shadow-sm">
            <div class="p-0">
                <div class="relative aspect-4/3 bg-muted">
                    <img
                        :src="imageUrl ?? undefined"
                        alt="Uploaded succulent photo"
                        decoding="async"
                        class="object-contain"
                        style="position: absolute; height: 100%; width: 100%; inset: 0px; color: transparent"
                    />
                </div>
            </div>
        </div>
        <div class="mb-6 flex items-start gap-2 rounded-lg bg-accent/50 p-3 text-sm">
            <TriangleAlert class="mt-0.5 h-4 w-4 shrink-0 text-primary" />

            <p class="text-muted-foreground">
                이 결과는 사진의 생김새를 바탕으로 예측한 것이며, 다른 종류일 가능성도 있어요. 각 결과를 눌러 자세한 정보를 확인해 보세요
            </p>
        </div>
        <div class="mb-8 space-y-3">
            <div
                class="group cursor-pointer rounded-lg border border-primary bg-primary/5 text-card-foreground shadow-sm transition-all duration-200 hover:bg-primary/10 hover:shadow-md"
                role="button"
                tabindex="0"
                aria-label="View details for Aeonium, ranked 1 with 78% confidence"
            >
                <div class="p-4 sm:p-5">
                    <div class="flex items-center gap-4">
                        <div
                            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary font-sans text-lg font-bold text-primary-foreground"
                        >
                            1
                        </div>
                        <div class="min-w-0 flex-1">
                            <div class="mb-1 flex items-center gap-2">
                                <h3 class="truncate font-serif text-xl font-medium">{{ results[0].ko }} ({{ results[0].en }})</h3>
                                <div
                                    class="inline-flex items-center rounded-full border border-transparent bg-primary px-2.5 py-0.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/80 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-none"
                                >
                                    Most likely
                                </div>
                            </div>
                            <div class="flex items-center gap-2">
                                <div class="h-2 max-w-30 flex-1 overflow-hidden rounded-full bg-muted">
                                    <div class="h-full rounded-full bg-primary transition-all" :style="`width: ${results[0].acc}%`"></div>
                                </div>
                                <span class="text-sm font-medium text-muted-foreground">{{ results[0].acc }}%</span>
                            </div>
                        </div>
                        <ChevronRight class="h-5 w-5 shrink-0 text-primary transition-transform group-hover:translate-x-1" />
                    </div>
                </div>
            </div>
            <div
                v-for="(item, index) in results.slice(1)"
                :key="item.en"
                class="group cursor-pointer rounded-lg border border-border bg-card text-card-foreground shadow-sm transition-all duration-200 hover:border-primary/50 hover:shadow-md"
                role="button"
                tabindex="0"
                aria-label="View details for Echeveria, ranked 2 with 48% confidence"
            >
                <div class="p-4 sm:p-5">
                    <div class="flex items-center gap-4">
                        <div
                            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-muted font-sans text-lg font-bold text-muted-foreground"
                        >
                            {{ index + 2 }}
                        </div>
                        <div class="min-w-0 flex-1">
                            <div class="mb-1 flex items-center gap-2">
                                <h3 class="truncate font-serif text-lg font-medium">{{ item.ko }} ({{ item.en }})</h3>
                            </div>
                            <div class="flex items-center gap-2">
                                <div class="h-2 max-w-30 flex-1 overflow-hidden rounded-full bg-muted">
                                    <div class="h-full rounded-full bg-muted-foreground/50 transition-all" :style="`width: ${item.acc}%`"></div>
                                </div>
                                <span class="text-sm font-medium text-muted-foreground">{{ item.acc }}%</span>
                            </div>
                        </div>
                        <ChevronRight class="h-5 w-5 shrink-0 text-primary transition-transform group-hover:translate-x-1" />
                    </div>
                </div>
            </div>
        </div>
        <div class="flex justify-center">
            <Link
                :href="home.url()"
                class="inline-flex h-11 items-center justify-center gap-2 rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:outline-none"
            >
                <RefreshCcw class="mr-2 h-4 w-4" />
                다른 사진으로 다시 시도하기
            </Link>
        </div>
    </div>
</template>
