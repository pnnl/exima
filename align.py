import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import seaborn as sns
import matplotlib.cm as cm
import os
import pandas as pd
import scipy
from matplotlib.offsetbox import AnchoredText

# Try loading image for first path in list.
# If it fails due to an empty list, simply return None
def load_image(path):

    if len(path) > 1:
        raise ValueError(f'More than one image found: {path}')
    elif len(path) == 0 or not os.path.exists(path[0]):
        # No image found for this case
        return None

    img = cv2.imread(path[0])
    return np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# Adapted from https://github.com/mwaskom/seaborn/issues/2280
def control_legend(ax, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, title=title, **kws)

def pval(m1, m2):
    return round(scipy.stats.ttest_ind(m1, m2)[1], 2)

def add_pvalues(ax, df, x_col, y_col, loc):
    # ax: to draw on
    # df: to pull data from
    # x_col: to be split into unique values
    # y_col: to perform stats on
    # loc: matplotlib style legend location string

    # Form text block
    text = 'P-values:'
    classes = df[x_col].unique()
    for i, a in enumerate(classes):
        temp_a = df[df[x_col] == a][y_col]  # Pull a values
        for b in classes[i + 1:]:
            temp_b = df[df[x_col] == b][y_col]  # Pull b values

            if len(temp_a) > 0 and len(temp_b) > 0:
                res = pval(temp_a, temp_b)  # Calculate P-value
                if res <= 0.001:
                    res = '<0.001'
            else:
                res = 'NaN'

            # Add result to text
            text += '\n'
            text += '   {} vs. {}: {}'.format(a, b, res)

    # Add to plot
    anc = AnchoredText(text, loc=loc, frameon=False)
    ax.add_artist(anc)

for experiment in [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24]:

    # Filepaths
    head_path = f'../data/Root Images/Experiment {experiment}/processed/'
    color_img_path = glob.glob(os.path.join(head_path, f'IMG_????.jpg'))
    color_membrane_img_path = glob.glob(os.path.join(head_path, f'*_soilstain.jpg'))
    root_mask_img_path = glob.glob(os.path.join(head_path, '*_traces.jpg'))
    nutrient_mask_path = glob.glob(os.path.join(head_path, '*chitin*.jpg'))

    for exudate in ['Phosphatase', 'Chitinase']:

        # Filepaths
        if exudate == 'Chitinase':
            membrane_stain_path = f'../data/Membrane Images/Chitinase {experiment}.tif'
        elif exudate == 'Phosphatase':
            membrane_stain_path = f'../data/Membrane Images/Phosphatase {experiment}.bmp'
        else:
            raise ValueError

        # Read in images (default is to read in as a BGR image)
        stain_img = load_image([membrane_stain_path])

        if stain_img is not None:
            output_path = f'/../output/Experiment {experiment}/{exudate}/'

            # Create output path if needed
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Load color image of rhizobox
            color_img = load_image(color_img_path)
            if color_img is None:
                raise RuntimeError('Missing required image: RGB photo of rhizobox')

            # Load color image of membrane overlaid on rhizobox
            color_membrane_img = load_image(color_membrane_img_path)
            if color_membrane_img is None:
                raise RuntimeError('Missing required image: RGB photo of membrane on rhizobox')

            # Load masks (nutrient band and roots)
            # If they do not exist, None is returned
            nutrient_mask_img = load_image(nutrient_mask_path)
            root_mask_img = load_image(root_mask_img_path)

            # Invert signal for stain
            # This will make higher signal -> higher value (max = 1)
            stain_img = np.invert(stain_img)

            # TODO: load these coords from a CSV
            # Define bounds of stain
            pts_color_rhizobox_all = {
                            1: [[492, 1352], [442, 1952], [126, 3702], [2151, 1406], [2184, 1868], [2472, 3702]],
                            2: [[610, 1371], [572, 1880], [258, 3771], [2404, 1396], [2369, 1798], [2639, 3778]],
                            3: [[650, 1545], [614, 1947], [272, 3744], [2298, 1582], [2279, 1893], [2610, 3773]],
                            4: [[568, 1308], [514, 1829], [182, 3752], [2459, 1317], [2501, 1886], [2723, 3729]],
                            5: [[387, 783], [407, 1253], [36, 3683], [2604, 754], [2622, 1163], [2938, 3627]],
                            6: [[431, 702], [412, 1334], [174, 3356], [2475, 738], [2459, 1243], [2615, 3374]],
                            7: [[523, 1015], [536, 1344], [267, 3581], [2502, 991], [2515, 1350], [2791, 3561]],
                            8: [[559, 1284], [540, 1800], [267, 3723], [2429, 1304], [2441, 1689], [2649, 3823]],
                            13: [[597, 1325], [560, 1751], [299, 3818], [2379, 1319], [2347, 1846], [2585, 3829]],
                            14: [[686, 1250], [678, 1645], [377, 3747], [2467, 1238], [2470, 1684], [2710, 3782]],
                            15: [[545, 895], [504, 1598], [225, 3709], [2588, 888], [2608, 1551], [2892, 3725]],
                            16: [[758, 1308], [730, 1753], [410, 3731], [2450, 1298], [2485, 1804], [2716, 3798]],
                            21: [[430, 533], [387, 1213], [175, 3512], [2600, 518], [2659, 1092], [3018, 3488]],
                            22: [[780, 1406], [782, 1883], [660, 3458], [2222, 1358], [2263, 1740], [2557, 3430]],
                            23: [[376, 927], [384, 1452], [162, 3783], [2398, 874], [2379, 1390], [2768, 3792]],
                            24: [[571, 900], [549, 1319], [357, 3575], [2416, 847], [2445, 1252], [2825, 3450]]
                            }

            #soil box with membrane 
            pts_color_membrane_rhizobox_edges_all = {
                            1: [[722, 1408], [676, 1994], [364, 3712], [2394, 1372], [2431, 1845], [2737, 3721]],
                            2: [[598, 1153], [570, 1683], [257, 3665], [2502, 1140], [2475, 1563], [2811, 3637]],
                            3: [[591, 1213], [547, 1680], [172, 3698], [2478, 1235], [2447, 1602], [2786, 3743]],
                            4: [[583, 997], [515, 1593], [125, 3761], [2742, 976], [2779, 1641], [2987, 3777]],
                            5: [[632, 909], [638, 1324], [240, 3526], [2580, 960], [2591, 1321], [2854, 3554]],
                            6: [[700, 1268], [690, 1798], [501, 3640], [2458, 1234], [2479, 1656], [2771, 3588]],
                            7: [[558, 918], [565, 1250], [250, 3499], [2561, 941], [2570, 1302], [2815, 3532]],
                            8: [[797, 1180], [785, 1572], [575, 3036], [2208, 1203], [2216, 1496], [2382, 3116]],
                            13: [[617, 1287], [590, 1713], [377, 3796], [2402, 1258], [2383, 1780], [2685, 3748]],
                            14: [[517, 1013], [514, 1458], [215, 3774], [2492, 995], [2495, 1494], [2750, 3788]],
                            15: [[582, 901], [543, 1613], [269, 3726], [2613, 910], [2631, 1576], [2899, 3753]],
                            16: [[490, 994], [478, 1507], [213, 3703], [2406, 953], [2456, 1521], [2760, 3678]],
                            21: [[624, 1126], [571, 1696], [317, 3646], [2454, 1144], [2489, 1632], [2748, 3694]],
                            22: [[438, 1114], [428, 1711], [237, 3620], [2292, 1118], [2330, 1587], [2635, 3615]],
                            23: [[726, 1434], [727, 1837], [532, 3636], [2287, 1437], [2271, 1831], [2541, 3682]],
                            24: [[531, 1169], [514, 1568], [342, 3556], [2242, 1201], [2243, 1582], [2466, 3503]]
                            }

            if exudate == 'Chitinase':

                # Edges of membrane in the color image
                pts_color_membrane_edges_all = {
                                1: [[1597, 2112], [1520, 2216], [1610, 3262], [2225, 2063], [2459, 3220]],
                                2: [[1675, 1863], [1580, 1940], [1577, 3124], [2369, 1897], [2485, 3170]],
                                3: [[1662, 1836], [1559, 1931], [1543, 3096], [2335, 1872], [2427, 3158]],
                                4: [[1793, 1883], [1655, 2002], [1647, 3398], [2571, 1921], [2708, 3426]],
                                5: [[902, 1537], [770, 1633], [720, 2817], [2404, 1582], [2558, 2847]],
                                6: [[926, 1972], [834, 2027], [782, 3187], [2290, 1971], [2464, 3165]],
                                7: [[885, 1567], [748, 1639], [677, 2866], [2388, 1586], [2564, 2886]],
                                8: [[995, 1760], [902, 1797], [837, 2717], [2111, 1774], [2217, 2745]],
                                13: [[854, 1964], [753, 2023], [711, 3182], [2221, 1962], [2346, 3177]],
                                14: [[848, 1737], [710, 1809], [611, 3087], [2293, 1719], [2380, 3074]],
                                15: [[905, 1774], [784, 1847], [720, 3197], [2415, 1790], [2574, 3208]],
                                16: [[790, 1702], [638, 1786], [542, 3024], [2226, 1683], [2328, 2993]],
                                21: [[1655, 1852], [1563, 1927], [1600, 3111], [2354, 1884], [2451, 3133]],
                                22: [[1490, 1822], [1390, 1901], [1432, 3042], [2153, 1815], [2293, 3018]],
                                23: [[1607, 1934], [1516, 1993], [1501, 2988], [2140, 1954], [2224, 2984]],
                                24: [[1575, 1750], [1450, 1838], [613, 2976], [2156, 1775], [2229, 2894]]
                                }

                # Edges of membrane in the stain image
                pts_stain_edges_all = {
                                1:[[1413, 437], [1573, 552], [1622, 2042], [617, 435], [569, 2032]],
                                2: [[1413, 359], [1590, 506], [1553, 1963], [551, 356], [587, 1972]],
                                3: [[1505, 411], [1636, 517], [1618, 2014], [604, 418], [602, 2037]],
                                4: [[440, 88], [498, 133], [498, 624], [178, 85], [156, 624]],
                                5: [[647, 87], [680, 140], [674, 643], [5, 87], [0, 638]],
                                6: [[1867, 373], [2046, 492], [2044, 2019], [33, 414], [17, 2045]],
                                7: [[622, 71], [675, 105], [679, 630], [11, 103], [9, 660]],
                                8: [[648, 110], [680, 135], [681, 656], [11, 122], [1, 666]],
                                13: [[1879, 358], [2021, 493], [2000, 1976], [8, 350], [2, 2003]],
                                14: [[1880, 382], [2045, 496], [2029, 2014], [14, 378], [1, 2012]],
                                15: [[1954, 282], [2046, 368], [2046, 1950], [32, 316], [2, 1955]],
                                16: [[1892, 289], [2047, 401], [2044, 1931], [9, 296], [28, 1960]],
                                21: [[461, 26], [523, 69], [501, 632], [119, 29], [130, 637]],
                                22: [[472, 68], [530, 103], [533, 672], [135, 83], [146, 680]],
                                23: [[1391, 405], [1562, 527], [1509, 1989], [553, 402], [498, 1991]],
                                24: [[469, 34], [533, 90], [538, 633], [155, 41], [161, 641]]
                                }

            elif exudate == 'Phosphatase':

                # Edges of membrane in the color image
                pts_color_membrane_edges_all = {
                                1: [[867, 2155], [769, 2287], [766, 3345], [1493, 2107], [1600, 3283]],
                                2: [[909, 1862], [783, 1971], [667, 3103], [1572, 1869], [1576, 3134]],
                                3: [[836, 1837], [751, 1929], [655, 3088], [1530, 1831], [1542, 3094]],
                                4: [[815, 1884], [715, 1956], [608, 3413], [1602, 1872], [1644, 3387]],
                                21: [[872, 1840], [786, 1898], [649, 3061], [1547, 1855], [1541, 3113]],
                                22: [[693, 1861], [566, 1967], [539, 3091], [1339, 1824], [1405, 3053]],
                                23: [[915, 1943], [833, 1992], [715, 3020], [1471, 1945], [1447, 3006]],
                                24: [[787, 1802], [710, 1853], [614, 2976], [1394, 1789], [1411, 2928]] 
                                }

                # Edges of membrane in the stain image
                pts_stain_edges_all = {
                                1: [[185, 88], [131, 161], [137, 724], [524, 93], [527, 736]],
                                2: [[208, 130], [150, 195], [146, 763], [542, 143], [540, 779]],
                                3: [[165, 105], [129, 160], [137, 742], [511, 111], [517, 751]],
                                4: [[228, 88], [190, 127], [194, 744], [561, 96], [571, 744]],
                                21: [[177, 132], [140, 166], [142, 753], [522, 141], [536, 769]],
                                22: [[195, 105], [137, 160], [139, 725], [523, 115], [515, 739]],
                                23: [[153, 71], [102, 100], [88, 707], [490, 76], [478, 702]],
                                24: [[186, 99], [147, 131], [137, 735], [517, 100], [517, 724]]
                                }

            # Pull points for this experiment and exudate
            pts_color_rhizobox = pts_color_rhizobox_all[experiment][:]
            pts_color_membrane_rhizobox_edges = pts_color_membrane_rhizobox_edges_all[experiment][:]
            pts_color_membrane_edges = pts_color_membrane_edges_all[experiment][:]
            pts_stain_edges = pts_stain_edges_all[experiment][:]

            # Flip stain horizontally to match root image orientation
            # Only needs to be done for chitinase images
            # TODO: fix coords so this doesn't need to be done for chitinase
            if exudate == 'Chitinase':
                stain_img = cv2.flip(stain_img, 1)
                pts_stain_edges = [[stain_img.shape[1] - x, y] for x, y in pts_stain_edges]

            ## Create figure comparing the point locations ##
            def plot_xform_pts(img, pts, h, w, subplot_num, title):
                plt.subplot(h, w, subplot_num)
                temp = img.copy()
                for pt in pts:
                    r = int(img.shape[0] / 50)
                    t = int(r / 5)
                    temp = cv2.circle(temp, pt, r, (255, 255, 255), t)
                    temp = cv2.circle(temp, pt, r - t, (0, 0, 0), -1)
                plt.imshow(temp)
                plt.title(title)
                plt.axis('off')

            fig, ax = plt.subplots(figsize = (6, 8))
            plot_xform_pts(color_img, pts_color_rhizobox, 2, 2, 1, 'Root Tracings')
            plot_xform_pts(color_membrane_img, pts_color_membrane_rhizobox_edges, 2, 2, 2, 'Box Edge Points')
            plot_xform_pts(color_membrane_img, pts_color_membrane_edges, 2, 2, 3, 'Membrane Edge Points')
            plot_xform_pts(stain_img, pts_stain_edges, 2, 2, 4, f'{exudate} Stain')

            plt.savefig(output_path + 'pts_prexform.png', dpi=400)
            plt.close(); plt.clf()

            ## Transform the color/root image to fit the color/membrane image ##

            # Calculate transform
            pts_color_rhizobox = np.float32(pts_color_rhizobox)
            pts_color_membrane_rhizobox_edges = np.float32(pts_color_membrane_rhizobox_edges)
            rows, cols = color_membrane_img.shape
            h, status = cv2.findHomography(pts_color_rhizobox, pts_color_membrane_rhizobox_edges)

            # Apply transform to soil image and derivatives
            color_img_xform = cv2.warpPerspective(color_img, h, (cols, rows))
            if root_mask_img is not None:
                root_mask_img_xform = cv2.warpPerspective(root_mask_img, h, (cols, rows))
            if nutrient_mask_img is not None:
                nutrient_mask_img_xform = cv2.warpPerspective(nutrient_mask_img, h, (cols, rows))

            # Soil + membrane image
            plt.subplot(1, 2, 1)
            plt.imshow(color_membrane_img)
            plt.title('Soil+Membrane Image')
            plt.axis('off')

            # Show transformed soil image
            plt.subplot(1, 2, 2)
            plt.imshow(color_img_xform)
            plt.title('Transformed Soil Image')
            plt.axis('off')

            plt.savefig(output_path + 'xform1.png', dpi=500, bbox_inches='tight')
            plt.close()
            plt.clf()

            ## Transform the soil image & soil + membrane images to fit the stain images ##

            # Calculate transform for stain
            pts_color_membrane_edges = np.float32(pts_color_membrane_edges)
            pts_stain_edges = np.float32(pts_stain_edges)
            rows, cols = stain_img.shape
            h, status = cv2.findHomography(pts_color_membrane_edges, pts_stain_edges)

            # Apply transform to soil image and derivatives
            color_img_xform2 = cv2.warpPerspective(color_img_xform, h, (cols, rows))
            if root_mask_img is not None:
                root_mask_img_xform2 = cv2.warpPerspective(root_mask_img_xform, h, (cols, rows))
            if nutrient_mask_img is not None:
                nutrient_mask_img_xform2 = cv2.warpPerspective(nutrient_mask_img_xform, h, (cols, rows))

            # Soil + membrane image
            plt.subplot(1, 2, 1)
            plt.imshow(stain_img)
            plt.title(f'{exudate} Signal')
            plt.axis('off')

            # Show transformed soil image
            plt.subplot(1, 2, 2)
            plt.imshow(color_img_xform2)
            plt.title('Transformed Soil Image')
            plt.axis('off')
            plt.savefig(output_path + 'xform2.png', dpi=500, bbox_inches='tight')
            plt.close()
            plt.clf()

            ## Crop images based on previous transform ##

            def crop_and_show(img, title, savename):
                img_crop = img[bounds[0]: bounds[1], bounds[2]: bounds[3]]
                plt.imshow(img_crop)
                plt.axis('off')
                plt.title(title)
                plt.savefig(output_path + savename, dpi=500, bbox_inches='tight')
                plt.close()
                plt.clf()

                return img_crop

            if exudate == 'Chitinase':
                bounds_all = {
                                1: [535, 2050, 460, 1450],
                                2: [500, 1965, 575, 1480],
                                3: [450, 2020, 470, 1450],
                                4: [133, 625, 175, 500],                
                                5: [85, 640, 5, 640],
                                6: [460, 2020, 30, 2020],
                                7: [105, 630, 15, 653],
                                8: [120, 630, 10, 660],
                                13: [490, 1980, 30, 1980],
                                14: [465, 1990, 10, 2030],
                                15: [350, 1960, 35, 2045],
                                16: [360, 1940, 25, 2020],
                                21: [50, 640, 180, 550],
                                22: [95, 675, 155, 535],
                                23: [510, 2000, 540, 1500],
                                24: [90, 635, 165, 510]
                                }
            elif exudate == 'Phosphatase':
                bounds_all = {
                                1: [110, 730, 150, 540],
                                2: [155, 760, 180, 545],
                                3: [140, 740, 150, 510],
                                4: [110, 740, 190, 560],                
                                21: [150, 750, 170, 525],
                                22: [140, 725, 140, 520],
                                23: [80, 700, 130, 480],
                                24: [110, 720, 170, 515]
                            }
            bounds = bounds_all[experiment]

            stain_img_crop = crop_and_show(stain_img, f'Cropped {exudate} Stain',
                                        'Stain_Cropped.png')
            color_img_xform2_crop = crop_and_show(color_img_xform2, 'Cropped Soil Image',
                                                'Soil_Cropped.png')

            if nutrient_mask_img is not None:
                nutrient_mask_img_xform2_crop = crop_and_show(nutrient_mask_img_xform2,
                                                            'Cropped Nutrient Band',
                                                            'NutrientBand_Cropped.png')

            if root_mask_img is not None:
                root_mask_img_xform2_crop = crop_and_show(root_mask_img_xform2, 'Cropped Root Mask',
                                                        'RootMask_Cropped.png')

            ## Create dataframe for all points ##

            # Initialize dataframe
            coords = np.where(stain_img_crop >= 0)
            df = pd.DataFrame(coords).T
            df = df.rename(columns={0:'i', 1:'j'})
            df['(i, j)'] = df[['i','j']].apply(tuple, axis=1)
            df['Root'] = False
            df['Nutrient Band'] = False
            df['Level'] = 'A'

            min_dist = 5
            max_dist = 10
            df['Close2Root'] = False
            df['Close2Nutrient'] = False
            df['Close2Both'] = False

            df['Dist2Root'] = np.nan
            df['Dist2Nutrient'] = np.nan
            df['Dist2Both'] = np.nan

            # Add stain image
            df['Stain'] = stain_img_crop.flatten()

            # Root related columns
            if root_mask_img is not None:
                mask = root_mask_img_xform2_crop > 252
                pixels = np.where(mask)
                pixels = list(zip(*pixels))
                df.loc[df['(i, j)'].isin(pixels), 'Root'] = True

                # Distance matrix: root
                mask = np.array(np.copy(mask), dtype=np.uint8)
                dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
                df['Dist2Root'] = dist.flatten()
                df['Close2Root'] = (df['Dist2Root'] > min_dist) & (df['Dist2Root'] <= max_dist)
            else:
                # Add dummy Dist2Root col
                df['Dist2Root'] = np.nan
                root_cutoff = 256

            # Nutrient related columns
            if nutrient_mask_img is not None:

                # Mask: Nutrient band
                mask = nutrient_mask_img_xform2_crop > 252
                pixels = np.where(mask)
                pixels = list(zip(*pixels))
                df.loc[df['(i, j)'].isin(pixels), 'Nutrient Band'] = True

                # Distance matrix: nutrient
                temp = np.array(np.copy(mask), dtype=np.uint8)
                dist = cv2.distanceTransform(1 - temp, cv2.DIST_L2, 5)
                df['Dist2Nutrient'] = dist.flatten()

                # Mask: pixels close to root and/or nutrient
                df['Close2Nutrient'] = (df['Dist2Nutrient'] > min_dist) & (df['Dist2Nutrient'] <= max_dist)

                # Create mask for pixels that are above the nutrient band
                above_mask = np.zeros(mask.shape, dtype=bool)
                for j in range(mask.shape[1]):

                    # Find middle pixel that is on the nutrient band (starting from the top)
                    i = np.where(mask[:, j])[0]
                    i = i[int(len(i) / 2)]

                    # Mark all pixels up to that point as above the nutrient band
                    above_mask[:i, j] = True
                above_mask[mask] = False  # Remove nutrient band from this mask
                above_mask = above_mask.flatten()

                # Finalize both masks
                df.loc[~above_mask, 'Level'] = 'C'
                df.loc[df['Nutrient Band'], 'Level'] = 'B'

            # Update columns with nutrient/root overlap
            df['Dist2Both'] = df[['Dist2Root', 'Dist2Nutrient']].apply(min, axis=1)
            df['Close2Both'] = df['Close2Root'] & df['Close2Nutrient']
            df['Both'] = df['Root'] & df['Nutrient Band']

            ## Plot all the things, as they appear in the df ##
            # Have to do this before removing or mixing any rows

            def plot_col_as_image(col, shape, title=None, savename=None, **kwargs):
                img = col.values.reshape(shape[0], shape[1])
                plt.imshow(img, interpolation='none', **kwargs)
                plt.axis('off')
                plt.title(title)

                if savename is not None:
                    plt.savefig(savename, dpi=400, bbox_inches='tight')

                plt.close()
                plt.clf()

            shape = stain_img_crop.shape

            plot_col_as_image(df['Stain'], shape, title='Stain',
                            savename=output_path + 'Stain')

            plot_col_as_image(df['Root'], shape, title='Roots',
                            savename=output_path + 'RootMask')
            plot_col_as_image(df['Dist2Root'], shape, title='Distance to Root',
                            savename=output_path + 'Dist2Root')
            plot_col_as_image(df['Close2Root'], shape, title='Close to Root',
                            savename=output_path + 'Close2Root')

            plot_col_as_image(df['Nutrient Band'], shape, title='Nutrient Band',
                            savename=output_path + 'NutrientBand')
            plot_col_as_image(df['Dist2Nutrient'], shape, title='Distance to Nutrient Band',
                            savename=output_path + 'Dist2Nutrient')
            plot_col_as_image(df['Close2Nutrient'], shape, title='Close to Nutrient Band',
                            savename=output_path + 'Close2Nutrient')

            plot_col_as_image(df['Both'], shape, title='Root & Nutrient Band Mask',
                            savename=output_path + 'Both')
            plot_col_as_image(df['Dist2Both'], shape, title='Distance to Root & Nutrient',
                            savename=output_path + 'Dist2Both')
            plot_col_as_image(df['Close2Both'], shape, title='Close to Root & Nutrient',
                            savename=output_path + 'Close2Both')

            ## Plot level labels ##
            mask = df['Level'].values.reshape(shape[0], shape[1])
            temp = np.zeros(shape)
            temp[mask == 'A'] = 2
            temp[mask == 'B'] = 1
            temp[mask == 'C'] = 0

            cmap = cm.get_cmap('viridis', 3)  # Select 3 discrete colors

            plt.imshow(temp, interpolation='none', cmap=cmap, vmin=0, vmax=2)
            plt.axis('off')
            plt.title('Level in Relation to Nutrient Band')

            cbar = plt.colorbar(ticks=np.arange(0.33, 2.1, 0.66), shrink=0.5, aspect=4)
            cbar.ax.yaxis.set_ticklabels(['Level C', 'Level B', 'Level A'], fontsize=12)
            cbar.ax.tick_params(size=0)

            plt.savefig(output_path + 'LevelMask', dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            ## Normalize stain values based on nutrient band ##
            # THIS IS WHY IT DOESN'T WORK

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

            # Plot raw stain values
            sns.histplot(data=df, x='Stain', y='Dist2Root', bins=20, ax=ax1, pthresh=0.05)
            ax1.set_xlabel('Signal (AU)')

            # Normalize stain values based on average nutrient band signal
            df['RawStain'] = df['Stain']  # Copy raw values to a new column
            if root_mask_img is None:  # No roots
                norm_values = df[df['Level'] == 'A']['Stain']
            else:
                norm_values = df[(df['Dist2Root'] >= 25) & (df['Level'] == 'A')]['Stain']
            df['Stain'] -= norm_values.mean()  # Set nutrient band mean to 0
            df['Stain'] /= norm_values.std()  # Set nutrient band std dev to 1

            # Plot normalized stain values
            sns.histplot(data=df, x='Stain', y='Dist2Root', bins=20, ax=ax2, pthresh=0.05)
            ax2.set_xlabel('Normalized Signal (AU)')
            ax2.set_ylabel(None)

            plt.savefig(output_path + 'StainValueNormalization', dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            if nutrient_mask_img is not None:
                plot_col_as_image(df['Stain'], shape, title='Normalized Stain',
                                vmin=-7, vmax=7,
                                savename=output_path + 'Stain_Normalized')

                # Since these are normalized, remove points >7 std away from the norm
                df = df[abs(df['Stain']) <= 7]

            df.to_pickle(output_path + 'df.pkl')

            # Split by classification and level with a new subplot for each level
            fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
            plt.subplots_adjust(wspace=0.2)
            for level, ax in zip(df['Level'].unique(), axes.flatten()):
                temp = df[df['Level'] == level]
                sns.barplot(data=temp, x='Root', y='RawStain', ax=ax, order=[False, True])
                
                ax.set_ylabel(None)

                # Each of these only has 2 classes
                classes = [False, True]
                x = temp[temp['Root'] == classes[0]]['RawStain']
                y = temp[temp['Root'] == classes[1]]['RawStain']
                if len(x) > 0 and len(y) > 0:
                    res = pval(x, y)
                    if res <= 0.001:
                        res = '<0.001'
                else:
                    res = 'NaN'
                ax.set_title('Level {} (P-value: {})'.format(level, res))

            axes[0].set_ylabel('Average Signal (AU)')
            plt.savefig(output_path + 'Barplot_Levels_Classes_Subplots_Pvalues',
                        dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            # Split by classification and level, all in one plot
            ax = sns.barplot(data=df, x='Level', y='RawStain', hue='Root', hue_order=[False, True])
            plt.ylabel('Average Signal (AU)')
            control_legend(ax, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(output_path + 'Barplot_Levels_Classes',
                        dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            ax = sns.boxplot(data=df, x='Level', y='RawStain', hue='Root', showfliers=False, hue_order=[False, True])
            plt.ylabel('Signal (AU)')
            control_legend(ax, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(output_path + 'Boxplot_Levels_Classes',
                        dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            # Compare across levels
            ax = sns.barplot(data=df, x='Level', y='RawStain')
            plt.ylabel('Average Signal (AU)')
            plt.savefig(output_path + 'Barplot_Levels',
                        dpi=400, bbox_inches='tight')
            add_pvalues(ax, df, 'Level', 'RawStain', 'upper center')
            plt.savefig(output_path + 'Barplot_Levels_Pvalues',
                        dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            ax = sns.boxplot(data=df, x='Level', y='RawStain', showfliers=False)
            plt.ylabel('Signal (AU)')
            plt.savefig(output_path + 'Boxplot_Levels',
                        dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            palette = ['#e41a1c', '#377eb8', '#4daf4a']
            temp_palette = [palette[0], palette[2], palette[1]]

            fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True, sharex=True)
            for ax, root in zip(axes.flatten(), [False, True]):
                
                temp = df.copy()
                temp.loc[temp['Level'] == 'A', 'Dist2Nutrient'] *= -1
                temp = temp[temp['Root'] == root]

                if len(temp) > 0:
                    sns.histplot(data=temp, x='Dist2Nutrient', y='RawStain', hue='Level',
                                hue_order=('A', 'C', 'B'), bins=21, palette=temp_palette, ax=ax,
                                legend=~root, pthresh=0.05)
                ax.set_xlabel('Distance to Nutrient Band (pixels)')
                
                ax.set_xticks([-500, -250, 0, 250, 500])
                ax.set_xticklabels([500, 250, 0, 250, 500])

                legend = ax.get_legend()
                if root:
                    ax.set_title('Root Pixels')
                    ax.set_ylabel(None)

                    if len(temp) > 0:  # Avoid errors
                        handles = legend.legendHandles
                        handles = [handles[0], handles[2], handles[1]]
                        legend.remove()
                        ax.legend(handles=handles, labels=['A', 'B', 'C'], title='Level', loc='upper right')
                        control_legend(ax, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                else:
                    ax.set_title('Soil Pixels')
                    ax.set_ylabel('Signal (AU)')
                    legend.remove()

            plt.savefig(output_path + f'Hist2D_Dist2Nutrient', dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()

            temp = df.copy()
            temp = temp[['(i, j)', 'RawStain', 'Dist2Root', 'Dist2Nutrient', 'Level']]
            temp['Dist2Root'] *= -1
            temp = temp.rename(columns={'Dist2Root': 'Root', 'Dist2Nutrient': 'Nutrient Band'})
            temp = temp.melt(id_vars=('(i, j)', 'Level', 'RawStain'), value_vars=('Root', 'Nutrient Band'), var_name='Distance to')
            temp = temp[abs(temp['value']) < 50]

            fig, axes = plt.subplots(3, 1, figsize=(5, 9), sharey=True, sharex=True)
            plt.subplots_adjust(hspace=0.2)
            for level, ax in zip(df['Level'].unique(), axes.flatten()):
                temp_level = temp[temp['Level'] == level]

                # palette = ['#e41a1c', '#377eb8', '#4daf4a'] # TODO: choose 3 other colors
                ax = sns.histplot(data=temp_level, x='value', y='RawStain', hue='Distance to', bins=20, ax=ax, pthresh=0.05)
                ax.set_title(f'Level {level}')
                plt.ylabel('Signal (AU)')
                control_legend(ax, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xticks([-50, -25, 0, 25, 50], [50, 25, 0, 25, 50])
            plt.xlabel('Distance (pixels)')

            plt.savefig(output_path + 'Hist2D_Dist2Both_Levels', dpi=400, bbox_inches='tight')
            plt.close()
            plt.clf()