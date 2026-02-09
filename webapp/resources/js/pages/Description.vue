<script setup lang="ts">
import { Link, router } from '@inertiajs/vue3';
import { ArrowLeft, Droplet, Flower2, Leaf } from 'lucide-vue-next';

import { findGenusByName } from '@/data/genera';
import { home } from '@/routes';

const props = defineProps<{
    genus: string;
}>();

const selectedGenus = findGenusByName(props.genus);

if (!selectedGenus) {
    router.visit(home.url());
}
</script>

<template>
    <div v-if="selectedGenus" class="mx-auto w-full max-w-2xl">
        <Link
            :href="home.url()"
            class="mb-4 -ml-2 inline-flex h-10 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium whitespace-nowrap text-muted-foreground ring-offset-background transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
        >
            <ArrowLeft class="mr-2 h-4 w-4" /> Back to results
        </Link>

        <h1 class="mb-6 font-serif text-3xl font-semibold text-foreground">{{ selectedGenus.displayName }}</h1>
        <div class="space-y-6">
            <div class="rounded-lg border bg-card text-card-foreground shadow-sm">
                <div class="p-6 pt-6">
                    <h3 class="mb-3 flex items-center gap-2 font-medium text-foreground"><Leaf class="h-4 w-4 text-primary" /> 주요 특징</h3>
                    <p class="leading-relaxed text-muted-foreground">
                        {{ selectedGenus.characteristics }}
                    </p>
                </div>
            </div>
            <div class="rounded-lg border bg-card text-card-foreground shadow-sm">
                <div class="p-6 pt-6">
                    <h3 class="mb-3 flex items-center gap-2 font-medium text-foreground">
                        <Droplet class="h-4 w-4 text-primary" />
                        키우는 법
                    </h3>
                    <p class="leading-relaxed text-muted-foreground">
                        {{ selectedGenus.careTips }}
                    </p>
                </div>
            </div>
            <div class="rounded-lg border bg-card text-card-foreground shadow-sm">
                <div class="p-6 pt-6">
                    <h3 class="mb-3 flex items-center gap-2 font-medium text-foreground">
                        <Flower2 class="h-4 w-4 text-primary" />
                        주요 품종
                    </h3>
                    <div class="flex flex-wrap gap-2">
                        <div
                            v-for="genus in selectedGenus.cultivars"
                            :key="genus"
                            class="inline-flex items-center rounded-full border border-transparent bg-secondary px-3 py-1.5 text-sm font-semibold text-secondary-foreground transition-colors hover:bg-secondary/80 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-none"
                        >
                            {{ genus }}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
