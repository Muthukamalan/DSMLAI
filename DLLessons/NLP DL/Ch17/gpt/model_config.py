model_args = {
    "init_args" : {
        "core" : {
            "optim":{
                "lr":2e-2,
                'weight_decay':0.99
            },
            "batch_size":1,
            "num_workers":0,
        },
        "block_size":1024,
        "vocab_size":50257,
        "n_layer" : 6,
        "n_head"  : 12,
        "n_embed" : 768,
        "dropout_rate":1e-1
    }
}