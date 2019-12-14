from vivid.featureset.molecules import create_molecule

from kaggle_days.atoms.basic import BasicCountEncoding, BasicOneHotEncoding, VersionsAtom, NumericAtom, GroupByUserAtom, \
    DateTimeAtom
from kaggle_days.atoms.dates import DateDiffAtom
from kaggle_days.atoms.encoding import TargetEncodingAtom
from kaggle_days.atoms.kiji import KijiCountAtom, KijiKeyowrdAtom, TextSWEMAtom, KijiGenreAtom, KijiDocVectorAtom, \
    KijiPublishTimeAtom

atoms = [
    BasicOneHotEncoding(),
    BasicCountEncoding(),
    NumericAtom(),
    VersionsAtom(),
    KijiCountAtom(),
    KijiKeyowrdAtom(),
    TextSWEMAtom(agg='mean', text_column='body'),
    TextSWEMAtom(agg='max', text_column='body'),
    TextSWEMAtom(agg='mean', text_column='title'),
    TextSWEMAtom(agg='max', text_column='title'),
    TextSWEMAtom(agg='mean', text_column='title2'),
    TextSWEMAtom(agg='max', text_column='title2'),
    TextSWEMAtom(agg='mean', text_column='keywords'),
    TextSWEMAtom(agg='max', text_column='keywords'),
    KijiGenreAtom(),
    DateTimeAtom()
]

m = create_molecule(name='benchmark', atoms=atoms)
user_merge_molecule = create_molecule(atoms=[GroupByUserAtom()], name='groupby_user')

create_molecule(name='add_kiji_vec', atoms=[
    *atoms,
    KijiDocVectorAtom()
])

create_molecule(name='target_encoding', atoms=[
    *atoms,
    KijiDocVectorAtom(),
    TargetEncodingAtom(use_columns=[
        'kiji_id',
        'er_dev_device_name',
        'er_dev_browser_family',
        'er_dev_device_type',
        'er_dev_os_family',
        'er_geo_city_j_name',
        'er_geo_country_code',
        'er_geo_pref_j_name',
        'ig_ctx_product',
        'ig_usr_connection'
    ])
])

create_molecule(name='add_time_diff', atoms=[
    *atoms,
    DateDiffAtom(),
    KijiDocVectorAtom(),
    KijiPublishTimeAtom(),
    TargetEncodingAtom(use_columns=[
        'kiji_id',
        'er_dev_device_name',
        'er_dev_browser_family',
        'er_dev_device_type',
        'er_dev_os_family',
        'er_geo_city_j_name',
        'er_geo_country_code',
        'er_geo_pref_j_name',
        'ig_ctx_product',
        'ig_usr_connection'
    ])
])
