<script setup lang="ts">
import { Link } from '@inertiajs/vue3';
import { ArrowLeft, Droplet, Flower2, Leaf } from 'lucide-vue-next';

import { home } from '@/routes';

const props = defineProps<{
    genus: string | null;
}>();

interface GenusInfo {
    id: string;
    name: string;
    displayName: string;
    characteristics: string;
    careTips: string;
    cultivars: string[];
}

const genusData: GenusInfo[] = [
    {
        id: 'aeonium',
        name: 'aeonium',
        displayName: '에오니움(Aeonium)',
        characteristics:
            '목질화된 줄기 위에 비교적 얇은 잎이 로제트 형태로 배열되어 자랍니다. 겨울에 성장하는 다육식물로, 여름 고온기에 매우 민감하여 잎을 오므리고 휴면에 들어갈 수 있습니다. 로제트 크기는 몇 cm에서 30cm 이상까지 다양합니다.',
        careTips:
            '여름에는 직사광선을 피하고 서늘하고 그늘진 곳에서 물을 거의 주지 않는 것이 좋습니다. 가을부터 봄까지의 생장기에는 충분한 빛과 함께 흙이 마르지 않도록 물을 주세요. 과습에 약하므로 배수가 잘되는 토양이 필수입니다.',
        cultivars: ['블랙로즈', '키위', '선버스트', '캐시미어 바이올렛'],
    },
    {
        id: 'dudleya',
        name: 'dudleya',
        displayName: '두들레야(Dudleya)',
        characteristics:
            '캘리포니아와 멕시코 원산으로, 잎 표면에 분가루처럼 보이는 흰 왁스층이 있는 단단한 로제트를 형성합니다. 극도로 건조에 강하며, 바위가 많은 배수 좋은 환경을 선호합니다. 은빛이나 청회색 계열의 색감을 지닌 종이 많습니다.',
        careTips:
            '잎의 왁스층이 손상되지 않도록 가급적 저면관수하세요. 배수가 매우 중요하며, 여름철 장마나 잦은 비를 피하는 것이 좋습니다. 습도가 높은 환경에서는 관리가 까다로울 수 있습니다.',
        cultivars: ['초크 두들레야', '브리틀부시', '캔들홀더', '캐니언 리브포에버'],
    },
    {
        id: 'echeveria',
        name: 'echeveria',
        displayName: '에케베리아(Echeveria)',
        characteristics:
            '가장 대중적인 다육식물 속 중 하나로, 두껍고 다육질의 잎이 대칭적인 로제트 형태를 이룹니다. 녹색, 청색, 분홍, 보라, 붉은색 등 다양한 색상을 가지며, 충분한 광량을 받으면 스트레스 컬러가 아름답게 발현됩니다.',
        careTips:
            '밝은 간접광이나 반양지를 선호합니다. 흙이 완전히 마른 뒤 충분히 물을 주세요. 로제트 중심부에 물이 고이지 않도록 주의하고, 마른 하엽은 주기적으로 제거해 주세요.',
        cultivars: ['엘레강스', '블랙 프린스', '롤라', '피코키'],
    },
    {
        id: 'graptopetalum',
        name: 'graptopetalum',
        displayName: '그랍토페탈룸(Graptopetalum)',
        characteristics:
            '일명 ‘고스트 플랜트’로 불리며, 두껍고 통통한 잎에 분가루가 덮인 경우가 많습니다. 성장하면서 줄기가 늘어져 늘어지는 형태를 띠어 행잉 플랜트로도 적합합니다. 잎 색은 회색, 분홍, 연보라 계열이 일반적입니다.',
        careTips:
            '건조와 방치에 비교적 강한 편입니다. 밝은 환경을 선호하지만 약간의 저광도에서도 적응합니다. 과습을 피하고 배수가 잘되는 환경을 유지하세요. 떨어진 잎으로도 번식이 잘 됩니다.',
        cultivars: ['고스트플랜트', '파라과이엔세', '브론즈', '수퍼붐'],
    },
    {
        id: 'haworthia',
        name: 'haworthia',
        displayName: '하월시아(Haworthia)',
        characteristics:
            '실내 재배에 적합한 소형 다육식물로, 잎 끝에 빛이 통과하는 반투명한 창(window) 구조를 가진 종이 많습니다. 단단한 로제트 형태 또는 뾰족한 잎이 쌓인 기둥 형태로 자랍니다.',
        careTips:
            '밝은 간접광에서 가장 잘 자라며, 강한 직사광선은 잎을 태울 수 있습니다. 흙이 마르면 물을 주고, 겨울에는 물 주는 횟수를 줄이세요. 비교적 낮은 광량에도 잘 적응합니다.',
        cultivars: ['제브라', '아테누아타', '쿠페리', '레투사', '심비포르미스'],
    },
    {
        id: 'lithops',
        name: 'lithops',
        displayName: '리톱스(Lithops)',
        characteristics:
            '‘살아있는 돌’이라 불리는 매우 독특한 다육식물로, 두 장의 잎이 하나로 붙어 있으며 중앙에 틈이 있습니다. 새로운 잎이 이 틈에서 자라며, 기존 잎은 서서히 말라 흡수됩니다.',
        careTips:
            '물 관리가 매우 중요합니다. 가을 개화 후와 헌 잎이 완전히 마른 뒤에만 물을 주세요. 여름 휴면기에는 절대 물을 주지 마세요. 배수가 극도로 좋은 토양과 충분한 햇빛이 필요합니다.',
        cultivars: ['레슬리', '카라스몬타나', '아우캄피에', '줄리', '옵티카 루브라'],
    },
    {
        id: 'pachyphytum',
        name: 'pachyphytum',
        displayName: '파키피툼(Pachyphytum)',
        characteristics:
            '달돌이나 콩처럼 보이는 매우 두껍고 통통한 잎이 특징입니다. 잎 표면에는 분가루가 덮여 있어 서리 낀 듯한 질감을 보이며, 짧은 줄기 위에 느슨한 로제트를 형성합니다.',
        careTips:
            '잎의 분가루가 쉽게 벗겨지므로 만질 때 주의하세요. 잎에 수분을 많이 저장하므로 물은 적게 주는 것이 좋습니다. 밝은 환경에서 색감이 좋아지며, 한여름 강한 햇빛은 피하세요.',
        cultivars: ['문스톤', '오비페룸', '콤팩툼', '글루티니카울레'],
    },
];

function getGenusByName(name: string | null): GenusInfo | undefined {
    if (name) {
        return genusData.find((g) => g.name === name.toLowerCase());
    }
}

const selectedGenus = getGenusByName(props.genus);
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
