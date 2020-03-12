# Fill in the datasets infos following the structure
# DATASETS = [
#     ('DATSET1_NAME', ('DATASET1_TRAIN_FILE', USE_TRAIN), ('DATASET1_VAL_FILE', USE_VAL),  'DATASET1_TEST_FILE'),
#     ('DATSET2_NAME', ('DATASET2_TRAIN_FILE', USE_TRAIN), ('DATASET2_VAL_FILE', USE_VAL),  'DATASET2_TEST_FILE'),
#     ...
#     ('DATSETN_NAME', ('DATASETN_TRAIN_FILE', USE_TRAIN), ('DATASETN_VAL_FILE', USE_VAL),  'DATASETN_TEST_FILE'),
# ]

DATASETS = [
    {'name': 'Macmorpho', 'trainFile': 'macmorpho-train.mm.txt', 'useTrain': True, 'valFile': 'macmorpho-dev.mm.txt',
        'useVal': True, 'testFile': 'macmorpho-test.mm.txt', 'tagSet': 'MM'},
    {'name': 'Bosque', 'trainFile': 'pt_bosque-ud-train.mm.txt', 'useTrain': True, 'valFile': 'pt_bosque-ud-dev.mm.txt',
        'useVal': True, 'testFile': 'pt_bosque-ud-test.mm.txt', 'tagSet': 'UD'},
    {'name': 'GSD', 'trainFile': 'pt_gsd-ud-train.mm.txt', 'useTrain': True, 'valFile': 'pt_gsd-ud-dev.mm.txt',
        'useVal': True, 'testFile': 'pt_gsd-ud-test.mm.txt', 'tagSet': 'UD'},
    {'name': 'Linguateca', 'trainFile': 'lgtc-train.mm.txt', 'useTrain': True, 'valFile': 'lgtc-dev.mm.txt',
        'useVal': True, 'testFile': 'lgtc-test.mm.txt', 'tagSet': 'LT'}
]

# Path to folder with datasets
DATASETS_DIR = 'data/'

###############################################################################################################################
