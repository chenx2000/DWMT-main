def set_template(args):
    # Set the templates here
    if args.template.find('dwmt') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'