from .GeoStreet_v1 import GEOSTREET_V1

__all__ = ['get_dataset']

def get_dataset(config):
    if config.DATASET.DATASET == 'GEOSTREET_V1':
        return GEOSTREET_V1
    else:
        raise NotImplemented()

