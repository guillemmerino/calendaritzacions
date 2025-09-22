import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import sys
from math import log2


from scipy.optimize import linear_sum_assignment
SCIPY_OK = True
#except Exception:
#    SCIPY_OK = False


# ---------- UTILITATS ----------

def parse_int(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        s = str(x).strip()
        if s.isdigit():
            return int(s)
        return default
    except Exception:
        return default

def obtenir_entitat(nom):
    # Deducció senzilla del nom de l'entitat (es pot adaptar segons el format real)
    import re
    return re.sub(r'\s+((["\']{1,2}).+?\2|[A-Za-zÀ-ÿ]+)$', '', str(nom)).strip()

def crear_grups_equilibrats(num_equips, max_grup=8, min_grup=6):
    '''
        Calcula el número de grups i el nombre d'equips per grup.
    '''

    # Si hi ha menys equips que el mínim, tot en un sol grup
    if num_equips < min_grup:
        return [num_equips]
    # Calcula quants equips per grup (mides equilibrades entre min i max)
    num_grups = max(1, (num_equips + max_grup - 1) // max_grup)
    while True:
        base = num_equips // num_grups
        sobra = num_equips % num_grups
        grups = [base + 1 if i < sobra else base for i in range(num_grups)]
        if max(grups) <= max_grup and min(grups) >= min_grup or num_grups >= num_equips:
            return grups
        num_grups += 1


# ---------- CONSTRUCCIÓ DE LA MATRIU DE COSTOS ----------
def build_slots(repartiment):
    '''
        Genera la llista de "slots" (grup, posició dins del grup). És a dir, genera una llista
        que guarda totes les posicions disponibles segons el repartiment (nº de grups i equips per grup)
        donat.
    '''
    slots_per_group = 8
    slots = []
    for g, size in enumerate(repartiment):
        for p in range(slots_per_group):
            slots.append((g, p))
    return slots

primera_fase = [
    [(8,5),(6,4), (7,3),(1,2)],
    [(2,8),(3,1), (4,7),(5,6)],
    [(8,6),(7,5), (1,4),(2,3)],
    [(3,8),(4,2), (5,1),(6,7)],
    [(8,7),(1,6), (2,5),(3,4)],
    [(8,4),(5,3), (6,2),(7,1)],
    [(1,8),(2,7), (3,6),(4,5)],
] 


def cost_calc(seed, g, p, disposicions, fase, w_dif_sorteig=3):

    preferits_casa = {8, 6, 7, 1}
    preferits_fora = {5, 4, 3, 2}
    cost = 0.
    if pd.isna(seed):
        cost = 0.0
    elif type(seed) == str and seed.strip().lower() == "fora":
        # p comença a 0, així que sumem 1 per comparar amb els valors preferits
        if (p + 1) in preferits_fora:
            cost = 0.0
        else:
            cost = 15.0  # Penalització alta per la resta

    elif type(seed) == str and seed.strip().lower() == "casa":
        if (p + 1) in preferits_casa:
            cost = 0.0
        else:
            cost = 15.0
    
    else:
        # Trobem la disposició casa, fora del seed
        seed_matches = []                
        for jornada in fase:
            for partit in jornada:
                # Revisem el seed
                if partit[0] == seed:
                    seed_matches.append("casa")
                if partit[1] == seed:
                    seed_matches.append("fora")
                                

        match = disposicions[p]
        difs = sum(a != b for a, b in zip(seed_matches, match))

        cost += w_dif_sorteig * difs
    return cost


def build_disposicions(fase):
    preferits_casa = {8, 6, 7, 1}
    preferits_fora = {5, 4, 3, 2}

    matches_0 = []
    matches_1 = []
    matches_2 = []
    matches_3 = []
    matches_4 = []
    matches_5 = []
    matches_6 = []
    matches_7 = []

    # Trobem la disposicio de partits casa, fora per cada equip
    for jornada in fase:
        for partit in jornada:
            # Cree la configuració de partits per equip
            if partit[0] == 1:
                matches_0.append("casa")
            if partit[1] == 1:
                matches_0.append("fora")
            if partit[0] == 2:
                matches_1.append("casa")
            if partit[1] == 2:
                matches_1.append("fora")
            if partit[0] == 3:
                matches_2.append("casa")
            if partit[1] == 3:
                matches_2.append("fora")
            if partit[0] == 4:
                matches_3.append("casa")
            if partit[1] == 4:
                matches_3.append("fora")
            if partit[0] == 5:
                matches_4.append("casa")
            if partit[1] == 5:
                matches_4.append("fora")
            if partit[0] == 6:
                matches_5.append("casa")
            if partit[1] == 6: 
                matches_5.append("fora")
            if partit[0] == 7:
                matches_6.append("casa")
            if partit[1] == 7:
                matches_6.append("fora")
            if partit[0] == 8:
                matches_7.append("casa")
            if partit[1] == 8:
                matches_7.append("fora")
                
        
    disposicions = [ matches_0, matches_1, matches_2, matches_3, matches_4, matches_5, matches_6, matches_7 ]


    return disposicions


def build_cost_matrix(df_cat, entity_costs, repartiment, w_entitat = 2.0, w_dif_sorteig=5, fase=primera_fase):
    """
    Matriu de costos (equips x slots) amb criteris:
    - proximitat al número de sorteig
    """



    num_grups = len(repartiment)
    slots = build_slots(repartiment) # Llista de (grup, posició)
    n_e = len(df_cat)   # Nombre d'equips
    n_s = len(slots)    # Nombre de slots (ha de ser igual a grups x 8)
    n_dummys = sum(8 - equips for equips in repartiment)
    C = np.zeros((n_e+n_dummys, n_s), dtype=float)

    df_cat = df_cat.copy()
    # Generem dins el df el nombre de dummys necessaris
    for n in range(n_dummys):
        df_cat.loc[n_e + n] = {
            'Núm. sorteig': np.nan,
            'Entitat': 'Dummy',
        }

    # Es crea una llista amb el número de sorteig de cada equip. Segueix l'ordre de df_cat
    seeds = df_cat['Núm. sorteig'].apply(
        lambda x: str(x).strip() if str(x).strip().lower() in ["fora", "casa"] else parse_int(x, default=np.nan)
    ).tolist()    
    entitats = df_cat['Entitat'].tolist()

    
    # Prepara una llista d'equips per entitat
    entitat_to_equips = defaultdict(list) # {entitat: [índexs equips]}
    for idx, ent in enumerate(entitats):
        entitat_to_equips[ent].append(idx)

    # Trobem la disposicio de partits casa, fora per cada equip
    disposicions = build_disposicions(fase)


    # Per cada num. sorteig (en ordre d'equips))
    vals = []
    if entity_costs:
        vals = [v for k, v in entity_costs.items() if k in set(entitats)]
    if vals:
        vmin, vmax = min(vals), max(vals)
        def norm(v):
            if vmax == vmin:
                return 1.0
            return (v - vmin) / (vmax - vmin)  # [0,1]
        
    
    for i, seed in enumerate(seeds):
        # i numera els equips.
        # Obtenim l'entitat de l'equip
        entitat = entitats[i]

        # Verifiquem si hi ha cost per entitat, sino, afegim l'entitat amb cost 0
        if entity_costs and entitat not in entity_costs:
            entity_costs[entitat] = 0

        # i el seu cost acumulat
        cost_entitat = entity_costs.get(entitat, 0) if entity_costs else 0
        factor_entitat = 1.0 + norm(cost_entitat) if vals else 1.0
        for j, (g, p) in enumerate(slots):
            # j númera els slots, combinacions úniques (grup, posició)
            cost = 0.

            cost = cost_calc(seed, g, p, disposicions, fase, w_dif_sorteig=w_dif_sorteig)

            C[i, j] = cost * factor_entitat

    return C, slots, entity_costs
'''
incorporar criteri perjidicacio rotatoria per entitat
Visio global de calendaris de totes les categories 
'''

def shannon_entropy(nivells):
    total = len(nivells)
    # Assignem valors numèrics als nivells
    nivells_numerics = []
    for nivell in nivells:
        if nivell == "Nivell A":
            nivells_numerics.append(1)
        elif nivell == "Nivell B":
            nivells_numerics.append(2)
        elif nivell == "Nivell C":
            nivells_numerics.append(3)
        elif nivell == "Nivell D":
            nivells_numerics.append(4)
        else:
            nivells_numerics.append(5)  # Per nivells desconeguts o buits
    if total == 0:
        return 0

    # Calculem arala suma de les diferències absolutes entre els nivells del grup
    entropia = 0.0
    for n1 in nivells_numerics:
        for n2 in nivells_numerics:
            if n1 != n2 and n1 < n2:
                entropia += abs(n1 - n2)

    return entropia



# ---------- CONTROL DE CONFLICTES PER ENTITAT ----------

def check_feasibility_entity(df_cat, repartiment):
    '''
        Comprova si hi ha alguna entitat amb més equips que grups
        Retorna un diccionari {entitat:count} de les entitats que no es poden separar.
        Exemple:
        Si tens 3 grups i el "Club X" té 4 equips, la funció retornarà {'Club X': 4}.
    '''
    num_grups = len(repartiment)
    counts = Counter(df_cat['Entitat']) # Contem nombre d'equips per entitat
    impossible = {e:c for e,c in counts.items() if c > num_grups} # Les fiquem al dict si hi ha més que grups
    return impossible


def build_groups_from_assignment(df_cat, slots, col_ind):
    '''
    Genera els grups a partir de l'assignació de slots. 
    La sortida és un diccionari {grup: [índexs equips]}, en ordre de 
    posició (numero sorteig) dins del grup.
    '''
    n_e_reals = len(df_cat)
    groups = defaultdict(dict)
    # Guardem (p, i_equ) per cada grup, només si i_equ < n_e_reals
    for i_equ, j_slot in enumerate(col_ind):
        if i_equ >= n_e_reals:
            continue  # Salta els dummys
        g, p = slots[int(j_slot)]
        groups[g][p] = i_equ
    return groups

def entity_conflicts(df_cat, groups):
    '''
    Comprova si hi ha conflictes d'entitat dins dels grups. 
    Retorna un diccionari {grup: {entitat:count}} de les entitats que tenen conflicte dins del grup.
    Exemple:
    Si en el grup 0 hi ha 2 equips del "Club X" i 1 del "Club Y", la funció retornarà {0: {'Club X': 2}}.
    '''
    conflicts = {}
    for g, pos_dict in groups.items():
        idxs = list(pos_dict.values())  # Ara pos_dict és {posició: índex_equip}
        ents = [df_cat.iloc[i]['Entitat'] for i in idxs]
        cnt = Counter(ents)
        over = {e: c for e, c in cnt.items() if c > 1 and e != 'Dummy'}
        if over:
            conflicts[g] = over
    return conflicts


def total_cost(C, row_ind, col_ind):
    return float(C[row_ind, col_ind].sum())



def repair_by_hungarian_per_position(df_cat, C, slots, row_ind, col_ind, conflicting_groups, groups,max_iters=10, penalty=1e6):
    """
    Per cada posició p conflictiva, resol una assignació òptima penalitzant molt si un grup ja té un equip de la mateixa entitat.
    """
    n_e = len(df_cat) # Nombre d'equips
    n_s = len(col_ind) # Nombre de slots
    entitats = df_cat['Entitat'].tolist()
    equip_to_slot = {i: col_ind[i] for i in range(n_e)} # {índex equip: índex slot}

    # Dins conflicting_groups hi ha els grups amb conflictes
    for conflict_group in conflicting_groups:
        # Obtenim el grup i identifiquem els equips de la mateixa entitat
        grup_teams = list(groups[conflict_group].values())
        entitats_in_group = [entitats[i] for i in grup_teams]
        entitats_repetides = {e for e, c in Counter(entitats_in_group).items() if c > 1}
        entitats_considerades = set()
        conflicting_ps = set()
        # Per cada equip del grup
        for i_equ in grup_teams:
            # Si la seva entitat està repetida, marquem la seva posició com conflictiva
            if entitats[i_equ] in entitats_repetides and entitats[i_equ] not in entitats_considerades:
                _, p = slots[int(equip_to_slot[i_equ])]
                conflicting_ps.add(p)
                entitats_considerades.add(entitats[i_equ])


        if not conflicting_ps:
            continue
        else:

            for p in conflicting_ps:
                #print(f"Reparant posició {p} del grup {conflict_group}...")
                # Obtenim els equips i slots de la posició p en els diferents grups
                equips_p = [i for i in range(n_e) if slots[equip_to_slot[i]][1] == p]
                slots_p = [j for j, (g, p2) in enumerate(slots) if p2 == p]

                if not equips_p or not slots_p:
                    continue

                # Construïm una matriu de costos local per aquests equips i slots
                C_local = np.zeros((len(equips_p), len(slots_p)))
                grups_entitats = defaultdict(set)
                for i in range(n_e):
                    g, p2 = slots[equip_to_slot[i]]
                    #if p2 == p:
                    grups_entitats[g].add(entitats[i])

                # Per cada equip de la posició p, i cada slot de la posició p, assigna cost
                for i_loc, i_equ in enumerate(equips_p):
                    entitat_i = entitats[i_equ]
                
                    for j_loc, j_slot in enumerate(slots_p):
                        g, _ = slots[j_slot]
                        # S'obté el cost previ de la matriu original
                        cost = C[i_equ, j_slot]
                        # Si el grup ja té aquesta entitat, penalitza molt
                        if entitat_i in grups_entitats[g]:
                            cost += penalty
                        C_local[i_loc, j_loc] = cost

                row_l, col_l = linear_sum_assignment(C_local)
                for i_loc, j_loc in zip(row_l, col_l):
                    i_equ = equips_p[i_loc]
                    j_slot = slots_p[j_loc]
                    equip_to_slot[i_equ] = j_slot

                # Reconstruim els grups segons la nova assignació per la següent iteració
                groups_actual = build_groups_from_assignment(df_cat, slots, [equip_to_slot[i] for i in range(n_e)])
                groups = groups_actual

    new_col_ind = np.array([equip_to_slot[i] for i in range(n_e)])
    return row_ind, new_col_ind



# ...existing code...

def homogeneitzar_nivell_costs(df_cat, groups, C, slots, fase=primera_fase, w_dif_sorteig=5, lambda_entropia=1.0, max_iters=3):
    """
    Millora local: intenta reduir (cost_sorteig + lambda_entropia * entropia_total)
    mitjançant swaps de posicions iguals (mateix número sorteig) entre grups diferents.
    Retorna (groups, C).
    """
    # Precalcula disposicions
    disposicions = build_disposicions(fase)

    # Funcions auxiliars
    def slot_idx(g, p):  # assumim 8 posicions per grup
        return g * 8 + p

    def entropia_grup(g, mapping):
        idxs = list(mapping[g].values())
        nivells = [df_cat.iloc[i]['Nivell'] for i in idxs]
        return shannon_entropy(nivells)

    # Entropia inicial per grup
    entropies = {g: entropia_grup(g, groups) for g in groups}
    entropia_total = sum(entropies.values())

    # Precompute seeds per equip
    seeds = df_cat['Núm. sorteig'].tolist()

    # Iteracions de millora local
    for _ in range(max_iters):
        millora = False
        # Llista de grups per ordre determinista
        grups_ids = sorted(groups.keys())
        for i_g, g1 in enumerate(grups_ids):
            for g2 in grups_ids[i_g+1:]:
                # intercanviar només posicions que existeixen als dos grups
                posicions_comunes = set(groups[g1].keys()) & set(groups[g2].keys())
                for p in sorted(posicions_comunes):
                    i_equ1 = groups[g1][p]
                    i_equ2 = groups[g2][p]

                    if i_equ1 == i_equ2:
                        continue

                    # Evita conflictes d'entitat
                    ent1 = df_cat.iloc[i_equ1]['Entitat']
                    ent2 = df_cat.iloc[i_equ2]['Entitat']
                    # Entitats presents sense els que es canvien
                    entitats_g1 = [df_cat.iloc[idx]['Entitat'] for pos, idx in groups[g1].items() if pos != p]
                    entitats_g2 = [df_cat.iloc[idx]['Entitat'] for pos, idx in groups[g2].items() if pos != p]
                    if ent2 in entitats_g1 or ent1 in entitats_g2:
                        continue

                    # Cost actual (només slots afectats + entropia)
                    s1 = slot_idx(g1, p)
                    s2 = slot_idx(g2, p)
                    cost_actual = C[i_equ1, s1] + C[i_equ2, s2] + lambda_entropia * entropia_total

                    # Recalcular costos si fem swap:
                    seed1 = seeds[i_equ1]
                    seed2 = seeds[i_equ2]
                    # Cost si i_equ1 va a (g2,p) i i_equ2 a (g1,p)
                    cost_i1_new = cost_calc(seed1, g2, p, disposicions, fase, w_dif_sorteig=w_dif_sorteig)
                    cost_i2_new = cost_calc(seed2, g1, p, disposicions, fase, w_dif_sorteig=w_dif_sorteig)

                    # Entropies noves
                    # Grup 1 després swap
                    nivells_g1_new = [df_cat.iloc[idx]['Nivell'] for pos, idx in groups[g1].items() if pos != p] + [df_cat.iloc[i_equ2]['Nivell']]
                    nivells_g2_new = [df_cat.iloc[idx]['Nivell'] for pos, idx in groups[g2].items() if pos != p] + [df_cat.iloc[i_equ1]['Nivell']]
                    ent_g1_new = shannon_entropy(nivells_g1_new)
                    ent_g2_new = shannon_entropy(nivells_g2_new)
                    entropia_total_new = entropia_total - entropies[g1] - entropies[g2] + ent_g1_new + ent_g2_new

                    nou_cost = cost_i1_new + cost_i2_new + lambda_entropia * entropia_total_new

                    if nou_cost < cost_actual:
                        # Accepta swap
                        groups[g1][p], groups[g2][p] = i_equ2, i_equ1
                        # Actualitza C només en les dues cel·les afectades
                        C[i_equ1, s2] = cost_i1_new  # nou slot
                        C[i_equ2, s1] = cost_i2_new
                        # (Opcional: podries posar a valor alt les antigues posicions,
                        # però ja no són usades per aquests equips)
                        entropies[g1] = ent_g1_new
                        entropies[g2] = ent_g2_new
                        entropia_total = entropia_total_new
                        millora = True
        if not millora:
            break

    return groups, C
# ...existing code...


def homogeneitzar_nivell(df_cat, groups, max_iters=100):
    """
    Optimitza la distribució de nivells entre grups minimitzant l'entropia,
    adaptat al format groups: {grup: {posició: índex_equip}}
    """
    for _ in range(max_iters):
        millora = False
        # Convertim els dicts de posicions a llistes per facilitar els swaps
        grups_llista = {g: list(pos_dict.items()) for g, pos_dict in groups.items()}
        for g1, items1 in grups_llista.items():
            for g2, items2 in grups_llista.items():
                if g1 >= g2:
                    continue
                for idx1, (p1, i1) in enumerate(items1):
                    for idx2, (p2, i2) in enumerate(items2):
                        if p1 != p2:
                            continue
                        # Comprova que el swap no genera conflicte d'entitat
                        ent1 = df_cat.iloc[i1]['Entitat']
                        ent2 = df_cat.iloc[i2]['Entitat']
                        ents1 = [df_cat.iloc[i]['Entitat'] for _, i in items1 if i != i1] + [ent2]
                        ents2 = [df_cat.iloc[i]['Entitat'] for _, i in items2 if i != i2] + [ent1]
                        if len(ents1) != len(set(ents1)) or len(ents2) != len(set(ents2)):
                            continue
                        # Calcula entropia abans i després
                        nivells1 = [df_cat.iloc[i]['Nivell'] for _, i in items1]
                        nivells2 = [df_cat.iloc[i]['Nivell'] for _, i in items2]
                        entropia_abans = shannon_entropy(nivells1) + shannon_entropy(nivells2)
                        nivells1_swap = [df_cat.iloc[i]['Nivell'] for _, i in items1 if i != i1] + [df_cat.iloc[i2]['Nivell']]
                        nivells2_swap = [df_cat.iloc[i]['Nivell'] for _, i in items2 if i != i2] + [df_cat.iloc[i1]['Nivell']]
                        entropia_despres = shannon_entropy(nivells1_swap) + shannon_entropy(nivells2_swap)
                        if entropia_despres < entropia_abans:
                            # Accepta el swap
                            items1[idx1] = (p1, i2)
                            items2[idx2] = (p2, i1)
                            millora = True
        if not millora:
            break
        # Reconstrueix groups amb el nou format
        groups = {g: {p: i for p, i in items} for g, items in grups_llista.items()}
    # Retorna el diccionari amb el nou format
    return groups


def actualitzar_costos_entitat(entity_costs, df_cat, C, row_ind, col_ind):
    
    '''
    Actualitza els costos per entitat segons l'assignació.
    '''
    costos_actualitzats = entity_costs.copy() if entity_costs else {}
    for r, c in zip(row_ind, col_ind):
        # Obtenim entitat del equips r
        if r >= len(df_cat):
            continue
        entitat = df_cat.iloc[r]['Entitat']
        # Obtenim el  cost actual
        if entitat != 'Dummy' and entity_costs and entitat in entity_costs:
            cost_entitat = entity_costs[entitat]
        else:
            cost_entitat = 0
        cost_rc = C[r, c]
        # Actualitzem el cost
        cost_actualitzat = cost_entitat + cost_rc 

        # Guardem el cost actualitzat
        costos_actualitzats[entitat] = cost_actualitzat

    return costos_actualitzats



# ---------- FUNCIÓ PRINCIPAL ----------

def assignar_grups_hungares(df_categoria, max_grup=8, min_grup=6, entity_costs=None, weights=None):
    """
    df_categoria ha de tenir com a mínim:
      - 'Nom', 'Nom Lliga', 'Núm. sorteig'
      - i opcionalment 'Entitat' (si no, es deriva de 'Nom')
    """
    df_cat = df_categoria.copy()
    if 'Entitat' not in df_cat.columns:
        df_cat['Entitat'] = df_cat['Nom'].apply(obtenir_entitat)

    repartiment = crear_grups_equilibrats(len(df_cat), max_grup=max_grup, min_grup=min_grup)
    impossible = check_feasibility_entity(df_cat, repartiment)

    # De moment, no resolem el cas impossible
    if impossible:
        raise ValueError(f"No és possible separar entitats: {impossible}")

    if weights is None:
        weights = dict(w_seed_group=5, w_seed_pos=1)
    C, slots, entity_costs = build_cost_matrix(df_cat, entity_costs=entity_costs, repartiment=repartiment, w_dif_sorteig=weights.get('w_dif_sorteig', 5) )
    #C_aug, bye_info = augment_with_byes_demanda(C, slots, repartiment, mega_penalty=1e9, epsilon=1e-3)

    # resolució hongaresa

    row_ind, col_ind = linear_sum_assignment(C)

    entity_costs = actualitzar_costos_entitat(entity_costs, df_cat, C, row_ind, col_ind)

    # Es creen els grups i es comproven conflictes
    g_assigned = build_groups_from_assignment(df_cat, slots, col_ind)
    conflicts = entity_conflicts(df_cat, g_assigned)
    #print("Conflictes inicials d'entitat:", conflicts)
    
    max_repair_iters = 10
    repair_iter = 0
    while conflicts and repair_iter < max_repair_iters:
        #print(f"Iteració de reparació {repair_iter+1}: Conflictes d'entitat: {conflicts}")
        
        #print("Conflictes per posició:", conflicts.keys())
        # Torna a intentar reparar
        row_ind, col_ind = repair_by_hungarian_per_position(
            df_cat, C, slots, row_ind=row_ind, col_ind=col_ind, conflicting_groups=conflicts, groups=g_assigned, max_iters=5
        )
        # Torna a construir grups i comprovar conflictes
        g_assigned = build_groups_from_assignment(df_cat, slots, col_ind)
        conflicts = entity_conflicts(df_cat, g_assigned)
        repair_iter += 1
        entity_costs = actualitzar_costos_entitat(entity_costs, df_cat, C, row_ind, col_ind)

    if repair_iter == max_repair_iters:
        print("AVÍS: No s'ha pogut resoldre tots els conflictes d'entitat després de diverses iteracions.")

    groups = build_groups_from_assignment(df_cat, slots, col_ind)
    
    # Actualitzem els cost de les entitats segons l'assignació final

    groups = homogeneitzar_nivell(df_cat, groups)
    groups, _ = homogeneitzar_nivell_costs(df_cat, groups, C, slots, w_dif_sorteig=5, lambda_entropia=1.0, max_iters=3)

    # calcula entropia de nivell
    nivell_entropia = {}
    for g, pos_dict in groups.items():
        idxs = list(pos_dict.values())
        nivells = [df_cat.iloc[i]['Nivell'] for i in idxs]
        nivell_entropia[g] = shannon_entropy(nivells)
        # Si vols imprimir l'entropia de cada grup individualment:
        #print(f"Entropia de nivell grup {g+1}: {nivell_entropia[g]}")
    entropia_total = sum(nivell_entropia.values())
    #print(f"Entropia total de nivell: {entropia_total:.3f}")

    # Afegim ara un refinament per dia de partit
# Després de l’últim bucle de reparació i abans del DataFrame:
    #col_ind = reoptimize_bye_within_groups(df_cat, C, slots, col_ind)


    

# crea el resultat
    assign = []
    diferencies_jornades = {}
    for g, pos_dict in sorted(groups.items()):
        for pos in sorted(pos_dict.keys()):
            i = pos_dict[pos]
            r = df_cat.iloc[i]

            # Comprovem jornades diferents
            seed = r['Núm. sorteig']
            # Disposició sol·licitada
            try:
                seed_int = int(seed)
            except Exception:
                diferencies_jornades[i] = []
                continue
            seed_matches = []
            for jornada in primera_fase:
                for partit in jornada:
                    if partit[0] == seed_int:
                        seed_matches.append("casa")
                    if partit[1] == seed_int:
                        seed_matches.append("fora")
            # Disposició assignada
            slot_matches = []
            for jornada in primera_fase:
                for partit in jornada:
                    if partit[0] == (pos + 1):
                        slot_matches.append("casa")
                    if partit[1] == (pos + 1):
                        slot_matches.append("fora")
            # Jornades on no coincideix
            dif_jornades = [j+1 for j, (a, b) in enumerate(zip(seed_matches, slot_matches)) if a != b]
            diferencies_jornades[i] = dif_jornades

    for g, pos_dict in sorted(groups.items()):
        for pos in range(8):
            if pos in pos_dict:
                i = pos_dict[pos]
                r = df_cat.iloc[i]
                assign.append({
                    'Nom Lliga': r['Nom Lliga'],
                    'Grup': f"G{g+1}",
                    'Nom': r['Nom'],
                    'Entitat': r['Entitat'],
                    'Nivell': r['Nivell'],
                    'Núm. sorteig': r['Núm. sorteig'],  # Sol·licitat
                    'Núm. sorteig assignat': pos + 1,    # Assignat realment
                    'Diferències jornades': diferencies_jornades[i],
                })
            else:
                # Escriu la fila buida o amb valors per slot buit
                assign.append({
                    'Nom Lliga': '',
                    'Grup': f"G{g+1}",
                    'Nom': '',
                    'Entitat': '',
                    'Nivell': '',
                    'Núm. sorteig': '',
                    'Núm. sorteig assignat': pos + 1,
                    'Diferències jornades': [],
                })
        res = pd.DataFrame(assign).sort_values(by=['Grup','Núm. sorteig assignat']).reset_index(drop=True)
   
   
   
    # comprova conflictes finals
    groups_final = defaultdict(list)
    for _, r in res.iterrows():
        groups_final[r['Grup']].append(r['Entitat'])
    conflicts_final = {g:dict(Counter(v)) for g,v in groups_final.items() if any(Counter(v).values())}
    conflicts_final = {g:{e:c for e,c in d.items() if c>1} for g,d in conflicts_final.items()}
    
    
    
    # --- VALIDACIÓ FINAL ---

    # 1. Comprova que no hi ha grups amb equips de la mateixa entitat
    conflicte_entitat = False
    for g, pos_dict in groups.items():
        idxs = list(pos_dict.values())
        ents = [df_cat.iloc[i]['Entitat'] for i in idxs]
        cnt = Counter(ents)
        if any(c > 1 for c in cnt.values()):
            print(f"AVÍS: El grup G{g+1} té més d'un equip de la mateixa entitat: {dict(cnt)}")
            conflicte_entitat = True

    # 2. Comprova que cada equip està assignat al número de sorteig sol·licitat
    sorteig_incorrecte = False
    for g, pos_dict in groups.items():
        for pos in sorted(pos_dict.keys()):
            i_equ = pos_dict[pos]
            sorteig_esperat = df_cat.iloc[i_equ]['Núm. sorteig']
            # Si és NaN, es considera correcte
            if pd.isna(sorteig_esperat):
                continue

            s = str(sorteig_esperat).strip().lower()
            # Cas especial "Fora"
            if s == "fora":
                if (pos + 1) in {5, 4, 3, 2}:
                    continue  # Correcte
                else:
                    print(f"AVÍS: L'equip '{df_cat.iloc[i_equ]['Nom']}' (grup G{g+1}) amb 'Fora' no està en una posició preferida ({pos+1})")
                    sorteig_incorrecte = True
                continue
            # Cas especial "Casa"
            if s == "casa":
                if (pos + 1) in {8, 6, 7, 1}:
                    continue  # Correcte
                else:
                    print(f"AVÍS: L'equip '{df_cat.iloc[i_equ]['Nom']}' (grup G{g+1}) amb 'Casa' no està en una posició preferida ({pos+1})")
                    sorteig_incorrecte = True
                continue

            # Si no és cap cas especial, intenta convertir a int
            try:
                sorteig_esperat_int = int(sorteig_esperat)
                if sorteig_esperat_int != (pos + 1):
                    print(f"AVÍS: L'equip '{df_cat.iloc[i_equ]['Nom']}' (grup G{g+1}) no té el número de sorteig assignat sol·licitat ({sorteig_esperat}), sinó {pos+1}")
                    sorteig_incorrecte = True
            except Exception:
                print(f"AVÍS: L'equip '{df_cat.iloc[i_equ]['Nom']}' (grup G{g+1}) té un número de sorteig invàlid: {sorteig_esperat}")
                sorteig_incorrecte = True

    if not conflicte_entitat and not sorteig_incorrecte:
        print("VALIDACIÓ: Assignació correcta. No hi ha conflictes d'entitat ni errors de sorteig.")
    if conflicte_entitat:
        print("VALIDACIÓ: S'han detectat conflictes d'entitat.")
    if sorteig_incorrecte:
        print("VALIDACIÓ: S'han detectat errors de sorteig.")
        

    return res, {
        'num_grups': len(repartiment),
        'repartiment': repartiment,
        'conflictes_entitat': conflicts_final,
        'categoria': df_cat.iloc[0]['Nom Lliga'],
    }
