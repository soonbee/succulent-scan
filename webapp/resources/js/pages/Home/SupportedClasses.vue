<script setup lang="ts">
const labels = [
    { ko: '에케베리아', en: 'echeveria' },
    { ko: '하월시아', en: 'haworthia' },
    { ko: '에오니움', en: 'aeonium' },
    { ko: '두들레야', en: 'dudleya' },
    { ko: '그랍토페탈롬', en: 'graptopetalum' },
    { ko: '리톱스', en: 'lithops' },
    { ko: '파키피텀', en: 'pachyphytum' },
];
</script>
<template>
    <div class="rounded-lg border border-border/60 bg-card text-card-foreground shadow-sm">
        <div class="flex flex-col space-y-1.5 p-6">
            <div class="flex items-center gap-2 text-base font-medium tracking-tight">지원하는 다육식물 종류</div>
            <div class="text-sm text-muted-foreground">현재 {{ labels.length }}가지 종류의 다육식물을 분류할 수 있어요</div>
        </div>
        <div class="p-6 pt-0">
            <div class="flex flex-wrap gap-2">
                <div
                    v-for="label in labels"
                    :key="label.en"
                    class="inline-flex items-center rounded-full border border-transparent bg-secondary px-3 py-1.5 text-sm font-semibold text-secondary-foreground transition-colors hover:bg-secondary/80 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-none"
                >
                    {{ label.ko }}(
                    <span class="capitalize">
                        {{ label.en }}
                    </span>
                    )
                </div>
            </div>
        </div>
    </div>
</template>
