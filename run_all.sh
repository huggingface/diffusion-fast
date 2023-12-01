#!/bin/bash

# Loop over compile_mode options
for compile_mode in reduce-overhead max-autotune; do
    # Loop over each boolean argument with two states: set and unset
    for enable_fused_projections in "" "--enable_fused_projections"; do
        for upcast_vae in "" "--upcast_vae"; do
            for compile_unet in "" "--compile_unet"; do
                for compile_vae in "" "--compile_vae"; do
                    for change_comp_config in "" "--change_comp_config"; do
                        for do_quant in "" "--do_quant"; do
                            # Construct the command
                            command="python run.py --compile_mode $compile_mode $enable_fused_projections $upcast_vae $compile_unet $compile_vae $change_comp_config $do_quant"
                            
                            # Run the command
                            echo "Running command: $command"
                            eval $command
                        done
                    done
                done
            done
        done
    done
done
