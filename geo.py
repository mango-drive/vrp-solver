import googlemaps
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gmaps = googlemaps.Client('')


def bounding_box(c, b_s):
    # c: center
    # bs: box size
    # from 8-ortools.py

    Location = namedtuple('Location', ['lat', 'lon'])

    #: Location[lat,lon]: the centre point of the area.
    (clat, clon) = Location(c[0], c[1])
    rad_earth = 6367  # km
    circ_earth = np.pi * rad_earth
    #: The lower left and upper right points
    extents = {
        'llcrnrlon': (clon - 180 * b_s /
                      (circ_earth * np.cos(np.deg2rad(clat)))),
        'llcrnrlat':
        clat - 180 * b_s / circ_earth,
        'urcrnrlon': (clon + 180 * b_s /
                      (circ_earth * np.cos(np.deg2rad(clat)))),
        'urcrnrlat':
        clat + 180 * b_s / circ_earth
    }

    return extents


def geocode(address):
    return gmaps.geocode(address)[0]['geometry']['location']


def reverse_geocode(l):
    return gmaps.reverse_geocode(l)


def geocode_df(df: pd.DataFrame, address_col='address'):
    df['gcode'] = df[address_col].apply(geocode)
    df['lat'] = [d['lat'] for d in df['gcode']]
    df['lng'] = [d['lng'] for d in df['gcode']]
    return df


def make_distance_mat(locs, method='haversine'):
    # from 8-ortools
    number = len(locs)
    distmat = np.zeros((number, number))
    methods = {'haversine': haversine}
    assert (method in methods)
    for frm_idx in range(number):
        for to_idx in range(number):
            if frm_idx != to_idx:
                frm_c = locs[frm_idx]
                to_c = locs[to_idx]
                distmat[frm_idx, to_idx] = haversine(
                    frm_c.lng, frm_c.lat, to_c.lng, to_c.lat
                )
    return (distmat)


def haversine(lon1, lat1, lon2, lat2):
    """
    from 8-ortools
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
    c = 2 * np.arcsin(np.sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km


def get_directions(s, e):
    return gmaps.directions(s, e, mode='bicycling')


def coord_str(coord):
    return ",".join([str(c) for c in coord])


def create_route(points):
    # get directions between each pair of points
    route = []
    lat = 'lat'
    lng = 'lng'
    for s, e in zip(points[::2], points[1::2]):
        start = ",".join([str(p) for p in s])
        end = ",".join([str(p) for p in e])
        directions = get_directions(start, end)
        steps = directions[0]['legs'][0]['steps']
        for step in steps:
            s, e = step['start_location'], step['end_location']
            l = [[s[lat], s[lng]], [s[lat], s[lng]]]

        route.extend(l)

    return route


def get_directions_polyline(start, end):
    directions = get_directions(start, end)
    polyline = directions[0]['overview_polyline']['points']
    return polyline


def get_polyline_route(points):
    route = []
    for start, end in zip(points[::2], points[1::2]):
        s, e = coord_str(start), coord_str(end)
        line = get_directions_polyline(s, e)
        route.append(line)
    return route


if __name__ == '__main__':
    start = '48.854779,2.350319'
    end = '48.8382763,2.3316999'

    polyline = get_directions_polyline(start, end)
    print(polyline)
