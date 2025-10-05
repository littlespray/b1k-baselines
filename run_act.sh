DATA_PATH=/root/workspace

datasets=(
    turning_on_radio
    # picking_up_trash
    # putting_away_Halloween_decorations
    # cleaning_up_plates_and_food
    # can_meat
    # setting_mousetraps
    # hiding_Easter_eggs
    # picking_up_toys
    # rearranging_kitchen_furniture
    # putting_up_Christmas_decorations_inside
    # set_up_a_coffee_station_in_your_kitchen
    # putting_dishes_away_after_cleaning
    # preparing_lunch_box
    # loading_the_car
    # carrying_in_groceries
    # bringing_in_wood
    # moving_boxes_to_storage
    # bringing_water
    # tidying_bedroom
    # outfit_a_basic_toolbox
    # sorting_vegetables
    # collecting_childrens_toys
    # putting_shoes_on_rack
    # boxing_books_up_for_storage
    # storing_food
    # clearing_food_from_table_into_fridge
    # assembling_gift_baskets
    # sorting_household_items
    # getting_organized_for_work
    # clean_up_your_desk
    # setting_the_fire
    # clean_boxing_gloves
    # wash_a_baseball_cap
    # wash_dog_toys
    # hanging_pictures
    # attach_a_camera_to_a_tripod
    # clean_a_patio
    # clean_a_trumpet
    # spraying_for_bugs
    # spraying_fruit_trees
    # make_microwave_popcorn
    # cook_cabbage
    # chop_an_onion
    # slicing_vegetables
    # chopping_wood
    # cook_hot_dogs
    # cook_bacon
    # freeze_pies
    # canning_food
    # make_pizza
)

python upload_huggingface.py
cd baselines/il_lib
python train.py data_dir=$DATA_PATH robot=r1pro task=behavior task.name=turning_on_radio arch=act eval_interval=9999999
