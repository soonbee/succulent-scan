import { readonly, ref } from 'vue';

const imageUrl = ref<string | null>(null);
const file = ref<File | null>(null);

export function useUploadedImage() {
    const setImage = (newFile: File) => {
        if (imageUrl.value) {
            URL.revokeObjectURL(imageUrl.value);
        }
        imageUrl.value = URL.createObjectURL(newFile);
        file.value = newFile;
    };

    const clear = () => {
        if (imageUrl.value) {
            URL.revokeObjectURL(imageUrl.value);
            imageUrl.value = null;
        }
        file.value = null;
    };

    return { imageUrl: readonly(imageUrl), file: readonly(file), setImage, clear };
}
