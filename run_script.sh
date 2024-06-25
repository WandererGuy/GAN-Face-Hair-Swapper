#!/bin/bash
rm -r swap_result_folder
mkdir swap_result_folder

target_folder_big="swap_source/output_real_life_GAN"

for target_folder in "$target_folder_big"/*
do 
    echo "$target_folder"
    # Define the folder containing the target image and source images
    # target_folder="swap_source/output_real_life_GAN/0.png"

    target_image="$target_folder/reference.png"

    # Create a directory for the results
    target_filename_without_extension=$(basename -- "$target_folder")
    target_folder_name="${target_filename_without_extension}_folder"
    echo "$target_folder_name"
    rm -r swap_result_folder/"$target_folder_name"

    mkdir "swap_result_folder/$target_folder_name"
    # Iterate over all source images in the target folder
    for source in "$target_folder"/*
    do
        source_filename=$(basename -- "$source")
        source_filename_without_extension="${source_filename%.*}"

        # Skip the target image itself
        if [ "$source_filename" == "reference.png" ]; then
            continue
        fi

        # Run the Python script for each source-target pair
        python 2_swap_image.py \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --use_source_segmentation \
        --num_seg 3 \
        --swap_index 17,18 \
        --show_grid True \
        --gan_face 0 \
        --bbox_modify 15 \
        --use_mask \
        --source_image "$source" \
        --pic_specific_path "$target_image" \
        --target_image "$target_image"
        wait

        # Create a subdirectory for each source-target pair and move the results
        result_subfolder_name="${source_filename_without_extension}_swap"
        mkdir "swap_result_folder/$target_folder_name/$result_subfolder_name"
        mv ALL_TEST_IMAGE "swap_result_folder/$target_folder_name/$result_subfolder_name"
    done
done 



# rm -r swap_result_folder
# mkdir swap_result_folder

# for i in '4601.jpg' '2689.jpg' '3531.jpg' '547.jpg' '4135.jpg' '4141.jpg' '2751.jpg' '2862.jpg' '6571.jpg' '5285.jpg' '2416.jpg' '320.jpg' '6400.jpg' '2705.jpg' '177.jpg' '466.jpg' '643.jpg' '3535.jpg' '6862.jpg' '2433.jpg' '2235.jpg' '168.jpg'
# do
#     filename_without_extension="${i%.*}"
#     addname="_folder"
#     namefolder="$filename_without_extension$addname"
#     python 2_swap_image.py \
#     --name people \
#     --Arc_path arcface_model/arcface_checkpoint.tar \
#     --use_source_segmentation \
#     --num_seg 3 \
#     --swap_index 17,18 \
#     --show_grid True \
#     --gan_face 0 \
#     --bbox_modify 15 \
#     --use_mask \
#     --source_image "swap_source/CelebA_HQ/fake_$filename_without_extension.PNG"  \
#     --pic_specific_path "swap_source/CelebA_HQ/$filename_without_extension.jpg"  \
#     --target_image "swap_source/CelebA_HQ/$filename_without_extension.jpg"
#     wait
#     mkdir "swap_result_folder/$namefolder"
#     mv ALL_TEST_IMAGE "swap_result_folder/$namefolder"
# done
