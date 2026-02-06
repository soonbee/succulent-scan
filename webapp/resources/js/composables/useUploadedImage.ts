import { readonly, ref } from 'vue';

const imageUrl = ref<string | null>(null);

export function useUploadedImage() {
    const setImage = (file: File) => {
        if (imageUrl.value) {
            URL.revokeObjectURL(imageUrl.value);
        }
        imageUrl.value = URL.createObjectURL(file);
    };

    const clear = () => {
        if (imageUrl.value) {
            URL.revokeObjectURL(imageUrl.value);
            imageUrl.value = null;
        }
    };

    return { imageUrl: readonly(imageUrl), setImage, clear };
}
