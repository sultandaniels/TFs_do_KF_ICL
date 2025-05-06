

def gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False):

    # Generate the checkpoint values to make predictions on
    #minval is the minimum value of the range
    #maxval is the maximum value of the range
    #train_int is the length of the training interval
    #phases is a list that contains the integers for which a new interval length starts

    # Initialize the list of checkpoint values
    ckpts = []

    full_range = range(minval, maxval, train_int)
    print("len(full_range)", len(full_range))

    phase_count = 0
    ckpt = 0

    scale = 0.25
    while ckpt < maxval:
        # print("pahse_count", phase_count)
        # print("ckpt", ckpt)

        if ckpt < phases[phase_count]:
            # print("multiplier", int(10**(scale*phase_count)))
            if hande_code_scale:
                mult = 2**phase_count
            else:
                mult = int(10**(scale*phase_count))
            # print("train_int*mult", train_int*mult)
            ckpt += train_int*mult
            ckpts.append(ckpt)
        else:
            phase_count += 1
            if hande_code_scale:
                mult = 2**phase_count
            else:
                mult = int(10**(scale*phase_count))
            # print("train_int*mult", train_int*mult)
            ckpt += train_int*mult
            ckpts.append(ckpt)

    return ckpts



if __name__ == "__main__":
    minval = 33000
    maxval = 60000
    train_int = 1000

    phases = [minval, 10000, 52000, maxval]

    ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
    print(ckpt_pred_steps)
    print(len(ckpt_pred_steps))
    



    #params for vanilla ident model:

    # minval = 100
    # maxval = 17600
    # train_int = 100
    # phases = [600, 800, 9600, maxval]
    # hande_code_scale = True

    #params for vanilla gauss model:

    # minval = 3000
    # maxval = 180000
    # train_int = 3000

    # phases = [3000, 90000, 135000, maxval]
    # hande_code_scale = False

