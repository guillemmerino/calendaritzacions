import pandas as pd
import os
import re
import numpy as np
from assignacions import assignar_grups_hungares


ENV = os.getenv("APP_ENV", "DEV")

if ENV == "DEV":
    BASE_PATH = os.getcwd()  # o una ruta fixa local
else:
    BASE_PATH = "/app/dades"  # ruta dins el contenidor en producció


def llegir_csv(nom_fitxer):
    try:
        df = pd.read_csv(nom_fitxer)
        print(df.head())  # Mostra les primeres 5 files
        return df
    except Exception as e:
        print(f"Error en llegir el fitxer: {e}")
        return None

def obtenir_entitat(nom):
        # Elimina l'espai i una sola lletra majúscula al final (ex: "Club Volei X A" -> "Club Volei X")
        return re.sub(r'\s+((["\']{1,2}).+?\2|[A-Za-zÀ-ÿ]+)$', '', nom)


def crear_grups_equilibrats(num_equips, max_grup=8):
    """
    Dona el nombre d'equips, retorna una llista amb el nombre d'equips per grup,
    procurant que tots els grups tinguin el nombre més igual possible d'equips,
    i cap grup tingui més de max_grup equips.
    """
    # Nombre mínim de grups necessaris
    num_grups = (num_equips + max_grup - 1) // max_grup

    while True:
        base = num_equips // num_grups
        sobra = num_equips % num_grups
        grups = [base + 1 if i < sobra else base for i in range(num_grups)]
        if max(grups) <= max_grup:
            return grups
        num_grups += 1


def processar_dades(df_copiat):
    df = df_copiat.copy()
    # Obtenim el número de lligues (categories) de la modalitat
    if 'Entitat' not in df.columns:
        df['Entitat'] = df['Nom'].apply(obtenir_entitat)

    categories = df['Nom Lliga'].unique()
    #print("Entitats úniques:", df['Entitat'].unique())

    # Processem cada categoria per separat
    for categoria in categories:
        df_categoria = df[df['Nom Lliga'] == categoria]

        # Calculem el nombre de equips, és a dir, registres
        num_equips = df_categoria.shape[0]

        # Revisem que no hi hagi dos equips amb el matex nom    
        noms_equips = df_categoria['Nom'].tolist()
        if len(noms_equips) != len(set(noms_equips)):
            raise ValueError(f"Error: Hi ha equips amb el mateix nom a la categoria {categoria}")
                
        print(f"Número d'equips a la categoria {categoria}: {num_equips}")

        # Trobem el número de grups per categoria. S'han de crear grups d'entre
        # 6 i 8 equips.
        repartiment = crear_grups_equilibrats(num_equips, max_grup=8)
        num_grups = len(repartiment)
        print(f"Repartiment d'equips per grup a la categoria {categoria}: {repartiment}")

        # Obtenim els possibles camps en num_sorteig
        camps_possibles = df_categoria['Núm. sorteig'].unique()
        #print(f"Camps possibles (Núm. sorteig) a la categoria {categoria}: {camps_possibles}")

        # Recodifiquem Num. sorteig en cas que sigui Nan, Fora o Casa
        df_categoria['Núm. sorteig'] = df_categoria['Núm. sorteig'].replace({'Nan': -1, 'Fora': -2, 'Casa': -3})

        # Resolem ara un problema d'optimizació de costos.
        # 1) Per cada equip de la categoria,
        nums_sorteig = []
        for equip in noms_equips:

            # 2) Obtenim el número de sorteig
            num_sorteig = df_categoria[df_categoria['Nom'] == equip]['Núm. sorteig'].values[0]
            if pd.isna(num_sorteig) or not str(num_sorteig).isdigit():
                num_sorteig = -1  # Assignem un valor neutre si no és vàlid
                
                # Si es fora o casa
                if str(num_sorteig).lower() in ['fora', 'casa']:
                    num_sorteig = -2 if str(num_sorteig).lower() == 'fora' else -3
            else:
                nums_sorteig.append(int(num_sorteig))
            # De moment, només fem cas als números enters. Filtrem els altres casos.
            # Matriu de costos: files = equips, columnes = grups
        cost_mateixa_entitat = 1000  # Cost molt alt si coincideixen entitats
        cost_diferencia_sorteig = 1  # Cost per unitat de diferència de sorteig

        # Inicialitzem la matriu de costos
        cost_matrix = np.zeros((num_equips, num_grups))

        # Per cada equip i cada grup
        for i in range(num_equips):
            for j in range(num_grups):
                # Cost base: diferència de sorteig amb la mitjana del grup (al principi, pots posar 0 o la mitjana global)
                cost = 0
                # Si vols, pots afinar-ho després amb la mitjana dels equips ja assignats al grup
                if nums_sorteig[i] != -1:
                    cost += cost_diferencia_sorteig * abs(nums_sorteig[i] - np.mean([n for n in nums_sorteig if n != -1]))
                # Cost extra si ja hi ha equips de la mateixa entitat al grup (al principi, no n'hi ha, però pots preparar-ho per l'algorisme d'assignació)
                # Aquí només prepares la matriu, l'algorisme d'assignació l'haurà de tenir en compte
                cost_matrix[i, j] = cost

        print("Matriu de costos inicial (sense entitats):")
        print(cost_matrix)





    print("Total d'equips:", df.shape[0])




def processar_dades_2(df):

    entity_costs = {}
    # Assegurem columnes mínimes
    cols_ok = {'Nom', 'Nom Lliga', 'Núm. sorteig'}
    missing = cols_ok - set(df.columns)
    if missing:
        raise ValueError(f"Falten columnes necessàries: {missing}")

    # Si no hi ha 'Entitat', la deduirem dins del mòdul
    categories = sorted(df['Nom Lliga'].dropna().unique())

    resultats_totals = []
    info_totals = []

    for categoria in categories:
        df_cat = df[df['Nom Lliga'] == categoria].copy()
        if df_cat.empty:
            continue

        try:
            # Pots ajustar max_grup/min_grup i pesos segons necessitats
            res_df, entity_costs, info = assignar_grups_hungares(
                df_cat,
                max_grup=8,
                min_grup=6,
                entity_costs=entity_costs,
                weights={'w_dif_sorteig': np.log2(27)}
            )
        except ValueError as e:
            # p.ex. una entitat té més equips que grups (no factible separar)
            print(f"[{categoria}] ERROR d'assignació: {e}")
            continue

        # Desa CSV per categoria
        safe = "".join(c for c in categoria if c.isalnum() or c in "._- ").strip().replace(" ", "_")
        out_path = os.path.join(BASE_PATH, f"assignacio_{safe}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[OK] {categoria} → {out_path}")
        print("Info:", info)

        resultats_totals.append(res_df.assign(_Categoria=categoria))
        info_totals.append({'categoria': categoria, **info})

    # (Opcional) Retorna un únic DataFrame concatenat amb totes les categories
    if resultats_totals:
        return pd.concat(resultats_totals, ignore_index=True), info_totals
    return pd.DataFrame(), info_totals

if __name__ == "__main__":
    # Llista tots els fitxers CSV del directori actual
    ruta_csv = os.path.join(BASE_PATH, "csv/")
    print("Ruta CSV:", ruta_csv)
    fitxers_csv = [f for f in os.listdir(ruta_csv) if f.endswith('.csv')]
    print("Fitxers CSV disponibles:")
    for idx, f in enumerate(fitxers_csv, 1):
        print(f"{idx}. {f}")

    try:
        num = int(input("Introdueix el número del fitxer CSV: "))
        if 1 <= num <= len(fitxers_csv):
            nom_fitxer = fitxers_csv[num - 1]
            df = llegir_csv(os.path.join(ruta_csv, nom_fitxer))
            if df is not None:
                processar_dades_2(df)
        else:
            print("Número fora de rang.")
    except ValueError:
        print("Has d'introduir un número vàlid.")