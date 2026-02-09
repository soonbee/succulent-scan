<script setup lang="ts">
import { Link } from '@inertiajs/vue3';

import { genusData } from '@/data/genera';
import { description } from '@/routes';
</script>
<template>
    <div class="rounded-lg border border-border/60 bg-card text-card-foreground shadow-sm">
        <div class="flex flex-col space-y-1.5 p-6">
            <div class="flex items-center gap-2 text-base font-medium tracking-tight">지원하는 다육식물 종류</div>
            <div class="text-sm text-muted-foreground">현재 {{ genusData.length }}가지 종류의 다육식물을 분류할 수 있어요</div>
        </div>
        <div class="p-6 pt-0">
            <div class="flex flex-wrap gap-2">
                <div
                    v-for="genus in genusData"
                    :key="genus.id"
                    class="inline-flex items-center rounded-full border border-transparent bg-secondary px-3 py-1.5 text-sm font-semibold text-secondary-foreground transition-colors hover:bg-secondary/80 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-none"
                >
                    <Link :href="description({ query: { genus: genus.name } }).url">
                        {{ genus.koreanName }}(
                        <span class="capitalize">
                            {{ genus.name }}
                        </span>
                        )
                    </Link>
                </div>
            </div>
        </div>
    </div>
</template>
