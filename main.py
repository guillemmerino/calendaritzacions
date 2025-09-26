import pandas as pd
import os
import re
import numpy as np
from assignacions import assignar_grups_hungares
import unicodedata, hashlib
# Helper per convertir índex de columna (0-based) a lletra Excel, amb fallback si no hi ha xlsxwriter
try:
    from xlsxwriter.utility import xl_col_to_name as _xl_col_to_name
    def _col_letter(idx: int) -> str:
        return _xl_col_to_name(idx)
except Exception:
    def _col_letter(idx: int) -> str:
        # Conversió manual 0-based -> 'A', 'B', ..., 'Z', 'AA', ...
        if idx < 0:
            return "A"
        letters = ""
        n = idx
        while n >= 0:
            n, rem = divmod(n, 26)
            letters = chr(65 + rem) + letters
            n -= 1
        return letters



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


def entitats_que_volen_casa(df):
    """
    Retorna { entitat: [llista_equips_que_han_demanat_casa] }.
    Accepta 'Casa' (case-insensitive)
    """
    dfx = df.copy()
    if 'Entitat' not in dfx.columns:
        # opcional: deduir entitat a partir del nom si cal
        dfx['Entitat'] = dfx['Nom'].apply(obtenir_entitat)

    s = dfx['Núm. sorteig']
    mask = s.astype(str).str.strip().str.lower().eq('casa')
    return (dfx[mask]
            .groupby('Entitat')['Nom']
            .apply(list)
            .to_dict())

def entitats_que_volen_fora(df):
    """
    Retorna { entitat: [llista_equips_que_han_demanat_fora] }.
    Accepta 'Fora' (case-insensitive)
    """
    dfx = df.copy()
    if 'Entitat' not in dfx.columns:
        # opcional: deduir entitat a partir del nom si cal
        dfx['Entitat'] = dfx['Nom'].apply(obtenir_entitat)

    s = dfx['Núm. sorteig']
    mask = s.astype(str).str.strip().str.lower().eq('fora')
    return (dfx[mask]
            .groupby('Entitat')['Nom']
            .apply(list)
            .to_dict())


def obtenir_entitat(nom):
    # Deducció senzilla del nom de l'entitat (es pot adaptar segons el format real)
    import re
    return re.sub(r'\s+((["\']{1,2}).+?\2|[A-Za-zÀ-ÿ]+)$', '', str(nom)).strip()

def _normalize_entity_name(name: str) -> str:
    # treu variacions d’accents/espais/majús-minus
    s = unicodedata.normalize('NFKC', str(name)).casefold().strip()
    s = " ".join(s.split())  # col·lapsa espais múltiples
    return s

def stable_slot_for_entity(entity_name: str, home_slots) -> int:
    key = _normalize_entity_name(entity_name)
    # hash estable independent de la sessió de Python
    h = hashlib.sha1(key.encode('utf-8')).hexdigest()
    num = int(h, 16)
    return home_slots[num % len(home_slots)]

def processar_dades_2(df):

    entity_costs = {}
    # Assegurem columnes mínimes
    cols_ok = {'Nom', 'Nom Lliga', 'Núm. sorteig'}
    missing = cols_ok - set(df.columns)
    if missing:
        raise ValueError(f"Falten columnes necessàries: {missing}")
    
    df = df.copy()
    if 'Entitat' not in df.columns:
        df['Entitat'] = df['Nom'].apply(obtenir_entitat)

    entitats_casa = entitats_que_volen_casa(df)
    # Usa llistes ordenades (no sets) per garantir estabilitat del mapping
    preferits_casa = [8, 6, 7, 1]

    entitats_fora = entitats_que_volen_fora(df)
    preferits_fora = [5, 4, 3, 2]

    entitats_to_slot_casa = {}
    for entitat, equips in entitats_casa.items():
        slot_casa = stable_slot_for_entity(entitat, home_slots=preferits_casa)
        if entitat not in entitats_to_slot_casa:
            entitats_to_slot_casa[entitat] = slot_casa
        elif entitats_to_slot_casa[entitat] != slot_casa:
            print(f"Atenció: entitat '{entitat}' té diferents slots de casa assignats ({entitats_to_slot_casa[entitat]} i {slot_casa})")

    entitats_to_slot_fora = {}
    for entitat, equips in entitats_fora.items():
        if entitat in entitats_to_slot_casa:
            continue
        slot_fora = stable_slot_for_entity(entitat, home_slots=preferits_fora)
        if entitat not in entitats_to_slot_fora:
            entitats_to_slot_fora[entitat] = slot_fora
        elif entitats_to_slot_fora[entitat] != slot_fora:
            print(f"Atenció: entitat '{entitat}' té diferents slots de fora assignats ({entitats_to_slot_fora[entitat]} i {slot_fora})")


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
                entitats_casa=entitats_to_slot_casa,
                entitats_fora=entitats_to_slot_fora,
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



    # --- PREPARA VALIDACIONS PER ESCRIURE A L'EXCEL (si hi ha resultats) ---
    df_val_count_summary = pd.DataFrame()
    df_val_count_by_cat = pd.DataFrame()
    df_val_entity_conflicts = pd.DataFrame(columns=["Categoria", "Grup", "Entitat", "Count"])
    df_val_casa_fora = pd.DataFrame(columns=["Entitat", "Categoria", "Equip", "Petició", "Esperat", "Assignat"])

    def _req_type(x):
        s = str(x).strip().casefold()
        if s == "casa":
            return "casa"
        if s == "fora":
            return "fora"
        return None

    def _is_real_team(row) -> bool:
        nom = row.get("Nom", "")
        ent = row.get("Entitat", "")
        if pd.isna(nom) or str(nom).strip() == "":
            return False
        if str(ent).strip() in ("", "Dummy"):
            return False
        return True

    contraris = {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4}

    if resultats_totals:
        all_results = pd.concat(resultats_totals, ignore_index=True)

        # 1) Comptatge
        input_total = len(df)
        assigned_real = all_results[all_results.apply(_is_real_team, axis=1)]
        assigned_total = len(assigned_real)
        status = "OK" if input_total == assigned_total else "KO"
        df_val_count_summary = pd.DataFrame([
            {"Mètrica": "Equips esperats (input)", "Valor": input_total},
            {"Mètrica": "Equips assignats (sense dummies)", "Valor": assigned_total},
            {"Mètrica": "Estat", "Valor": status},
        ])
        per_cat_in = df.groupby("Nom Lliga").size().rename("Esperats").to_frame()
        per_cat_assigned = (
            assigned_real.groupby("Nom Lliga").size().rename("Assignats").to_frame()
            if not assigned_real.empty else pd.DataFrame(columns=["Assignats"])  # empty
        )
        df_val_count_by_cat = per_cat_in.join(per_cat_assigned, how="outer").fillna(0).reset_index().rename(columns={"Nom Lliga": "Categoria"})
        df_val_count_by_cat["Esperats"] = df_val_count_by_cat["Esperats"].astype(int)
        df_val_count_by_cat["Assignats"] = df_val_count_by_cat["Assignats"].astype(int)
        df_val_count_by_cat["OK"] = df_val_count_by_cat["Esperats"] == df_val_count_by_cat["Assignats"]

        # 2) Conflictes d'entitat per grup
        rows_conf = []
        for cat, df_cat_res in all_results.groupby("Nom Lliga"):
            for grup, df_grup in df_cat_res.groupby("Grup"):
                ents = [e for e in df_grup["Entitat"].tolist() if e and e != "Dummy"]
                if not ents:
                    continue
                cnt = pd.Series(ents).value_counts()
                for entitat, c in cnt.items():
                    if c > 1:
                        rows_conf.append({"Categoria": cat, "Grup": grup, "Entitat": entitat, "Count": int(c)})
        if rows_conf:
            df_val_entity_conflicts = pd.DataFrame(rows_conf)

        # 3) Coherència CASA/FORA per entitats
        incoherencies = []
        def _check_entity_consistency(entitat: str, slot_casa: int | None, slot_fora: int | None, priority_casa: bool):
            sub = all_results[all_results["Entitat"] == entitat].copy()
            if sub.empty:
                return
            sub = sub[sub.apply(_is_real_team, axis=1)]
            if sub.empty:
                return
            for _, r in sub.iterrows():
                req = _req_type(r.get("Núm. sorteig"))
                if req is None:
                    continue
                assigned = int(r.get("Núm. sorteig assignat", 0))
                if priority_casa:
                    if req == "casa" and slot_casa is not None and assigned != slot_casa:
                        incoherencies.append([entitat, r["Nom Lliga"], r["Nom"], "casa", slot_casa, assigned])
                    if req == "fora" and slot_casa is not None and assigned != contraris.get(slot_casa):
                        incoherencies.append([entitat, r["Nom Lliga"], r["Nom"], "fora", contraris.get(slot_casa), assigned])
                else:
                    if req == "fora" and slot_fora is not None and assigned != slot_fora:
                        incoherencies.append([entitat, r["Nom Lliga"], r["Nom"], "fora", slot_fora, assigned])
                    if req == "casa" and slot_fora is not None and assigned != contraris.get(slot_fora):
                        incoherencies.append([entitat, r["Nom Lliga"], r["Nom"], "casa", contraris.get(slot_fora), assigned])

        # Dicc. definits més amunt: entitats_to_slot_casa, entitats_to_slot_fora
        for entitat, slot_casa in entitats_to_slot_casa.items():
            _check_entity_consistency(entitat, slot_casa=slot_casa, slot_fora=None, priority_casa=True)
        for entitat, slot_fora in entitats_to_slot_fora.items():
            if entitat in entitats_to_slot_casa:
                continue
            _check_entity_consistency(entitat, slot_casa=None, slot_fora=slot_fora, priority_casa=False)

        if incoherencies:
            df_val_casa_fora = pd.DataFrame(incoherencies, columns=["Entitat", "Categoria", "Equip", "Petició", "Esperat", "Assignat"])


    # --- DESPRÉS DEL BUCLE PER CATEGORIES ---
    # Escriu tot en un sol Excel amb format
    excel_path = os.path.join(BASE_PATH, "assignacions.xlsx")
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Formats
        fmt_header = workbook.add_format({
            "bold": True, "align": "center", "valign": "vcenter",
            "bg_color": "#1F4E78", "font_color": "white", "border": 1
        })
        fmt_title = workbook.add_format({
            "bold": True, "font_size": 14, "align": "left", "valign": "vcenter"
        })
        fmt_default = workbook.add_format({"border": 1})
        fmt_wrap = workbook.add_format({"text_wrap": True, "border": 1})
        fmt_group_colors = {
            # ajusta colors si vols (8 → blau, 7 → verd, 6 → taronja, 5 → gris, etc.)
            1: workbook.add_format({"bg_color": "#E2EFDA"}),  # verd clar
            2: workbook.add_format({"bg_color": "#FFF2CC"}),  # groc clar
            3: workbook.add_format({"bg_color": "#FCE4D6"}),  # salmó
            4: workbook.add_format({"bg_color": "#E7E6E6"}),  # gris clar
            5: workbook.add_format({"bg_color": "#DDEBF7"}),  # blau clar
            6: workbook.add_format({"bg_color": "#E2EFDA"}),
            7: workbook.add_format({"bg_color": "#FFF2CC"}),
            8: workbook.add_format({"bg_color": "#FCE4D6"}),
        }

        # --- FULL "Resum" opcional amb info agregada ---
        used_sheet_names = set()
        if info_totals:
            df_info = pd.DataFrame(info_totals)
            df_info.to_excel(writer, sheet_name="Resum", index=False)
            ws_info = writer.sheets["Resum"]
            used_sheet_names.add("Resum")
            # capçalera
            for col_idx, _ in enumerate(df_info.columns):
                ws_info.write(0, col_idx, df_info.columns[col_idx], fmt_header)
                ws_info.set_column(col_idx, col_idx, 18)
            ws_info.autofilter(0, 0, max(0, len(df_info)), max(0, len(df_info.columns) - 1))
            ws_info.freeze_panes(1, 0)

            # Escriu seccions de VALIDACIONS a continuació
            start_row = len(df_info) + 2
            ws_info.write(start_row, 0, "VALIDACIONS", fmt_title)
            start_row += 2

            # Comptatge
            if not df_val_count_summary.empty:
                ws_info.write(start_row, 0, "Recompte global", fmt_header)
                df_val_count_summary.to_excel(writer, sheet_name="Resum", index=False, startrow=start_row+1)
                start_row = start_row + 2 + len(df_val_count_summary)

            if not df_val_count_by_cat.empty:
                start_row += 1
                ws_info.write(start_row, 0, "Recompte per categoria", fmt_header)
                df_val_count_by_cat.to_excel(writer, sheet_name="Resum", index=False, startrow=start_row+1)
                start_row = start_row + 2 + len(df_val_count_by_cat)

            # Conflictes d'entitat
            if not df_val_entity_conflicts.empty:
                start_row += 1
                ws_info.write(start_row, 0, "Conflictes d'entitat per grup", fmt_header)
                df_val_entity_conflicts.to_excel(writer, sheet_name="Resum", index=False, startrow=start_row+1)
                start_row = start_row + 2 + len(df_val_entity_conflicts)
            else:
                start_row += 1
                ws_info.write(start_row, 0, "Conflictes d'entitat per grup", fmt_header)
                ws_info.write(start_row+1, 0, "Cap conflicte detectat.")
                start_row += 3

            # CASA/FORA incoherències
            if not df_val_casa_fora.empty:
                ws_info.write(start_row, 0, "CASA/FORA – incoherències", fmt_header)
                df_val_casa_fora.to_excel(writer, sheet_name="Resum", index=False, startrow=start_row+1)
                start_row = start_row + 2 + len(df_val_casa_fora)
            else:
                ws_info.write(start_row, 0, "CASA/FORA – incoherències", fmt_header)
                ws_info.write(start_row+1, 0, "Cap incoherència detectada.")
                start_row += 3

        # --- FULL per categoria ---
        for res_df_cat in resultats_totals:
            categoria = res_df_cat["_Categoria"].iloc[0] if "_Categoria" in res_df_cat.columns else "Categoria"
            base = "".join(c for c in categoria if c.isalnum() or c in "._- ").strip().replace(" ", "_") or "Categoria"
            # Garanteix unicitat i límit de 31 caràcters
            sheet_name = base[:31]
            if sheet_name in used_sheet_names:
                # Afegeix sufix _2, _3... mantenint límit 31
                i = 2
                while True:
                    suffix = f"_{i}"
                    candidate = (base[: (31 - len(suffix))] + suffix)
                    if candidate not in used_sheet_names:
                        sheet_name = candidate
                        break
                    i += 1
            used_sheet_names.add(sheet_name)

            # copia i ordena per “Grup” i dins de cada grup per “Núm. sorteig assignat” si existeixen
            df = res_df_cat.drop(columns=[c for c in ["_Categoria"] if c in res_df_cat.columns]).copy()
            if "Grup" in df.columns and "Núm. sorteig assignat" in df.columns:
                df.sort_values(["Grup", "Núm. sorteig assignat"], inplace=True, kind="stable")
            elif "Grup" in df.columns:
                df.sort_values(["Grup"], inplace=True, kind="stable")

            # escriu a partir de fila 1 (deixem fila 0 per al títol gran)
            start_row = 1
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
            ws = writer.sheets[sheet_name]

            n_rows, n_cols = df.shape

            # Títol a la fila 0
            ws.merge_range(0, 0, 0, max(0, n_cols - 1), f"Assignació – {categoria}", fmt_title)

            # capçaleres amb format
            for col_idx, col_name in enumerate(df.columns):
                ws.write(start_row, col_idx, col_name, fmt_header)

            # vora per defecte + autoajust aproximat
            for col_idx, col_name in enumerate(df.columns):
                # amplada segons el màxim entre header i dades (simple aproximació)
                max_len = max(
                    len(str(col_name)),
                    *(len(str(x)) for x in df[col_name].head(200))  # no escanegem tot per rendiment
                )
                ws.set_column(col_idx, col_idx, min(40, max(12, max_len + 2)))  # límit 40

            # congela panell sota capçalera
            ws.freeze_panes(start_row + 1, 0)

            # filtres
            ws.autofilter(start_row, 0, start_row + max(0, n_rows), max(0, n_cols - 1))

            # text wrap per columnes llargues (opcional)
            for col_idx, col_name in enumerate(df.columns):
                if any(isinstance(x, str) and len(x) > 40 for x in df[col_name].head(100)):
                    ws.set_column(col_idx, col_idx, None, fmt_wrap)

            # condicional per color de “Grup” (si existeix) – pinta la fila sencera per cada grup
            if n_rows > 0 and "Grup" in df.columns:
                grup_col_idx = df.columns.get_loc("Grup")
                col_letter = _col_letter(grup_col_idx)  # p.ex. 'B'
                # recorrem els grups únics i apliquem format per fórmula (comparant text "G1", "G2", ...)
                for g in sorted(df["Grup"].dropna().astype(str).unique()):
                    g_str = str(g).strip()
                    # extreu número final per escollir color, si existeix
                    m = re.search(r"(\d+)$", g_str)
                    g_num = int(m.group(1)) if m else None
                    fmt = fmt_group_colors.get(g_num)
                    if not fmt:
                        continue
                    first_data_row = start_row + 1
                    last_data_row = start_row + n_rows
                    # compara el text del grup de la fila
                    ws.conditional_format(
                        first_data_row, 0, last_data_row, max(0, n_cols - 1),
                        {
                            "type": "formula",
                            "criteria": f'=${col_letter}{first_data_row+1}="{g_str}"',
                            "format": fmt           
                        }
                    )
                    # Truc: la fórmula s’avalua per cada fila; per assegurar que miri la fila correcta,
                    # xlsxwriter substitueix el número de fila segons la cel·la d’avaluació.

            # línies de taula (vora fina) a totes les cel·les
            for r in range(start_row + 1, start_row + 1 + n_rows):
                ws.set_row(r, None, fmt_default)

        # missatge final
    print(f"[OK] Excel generat → {excel_path}")

    # --- VALIDACIONS GLOBALS POST-PROCÉS ---
    def _req_type(x):
        s = str(x).strip().casefold()
        if s == "casa":
            return "casa"
        if s == "fora":
            return "fora"
        return None

    def _is_real_team(row) -> bool:
        nom = row.get("Nom", "")
        ent = row.get("Entitat", "")
        if pd.isna(nom) or str(nom).strip() == "":
            return False
        if str(ent).strip() in ("", "Dummy"):
            return False
        return True

    contraris = {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4}

    if resultats_totals:
        all_results = pd.concat(resultats_totals, ignore_index=True)

        # 1) Verificació de recompte d'equips (excloent dummies i files buides)
        input_total = len(df)
        assigned_real = all_results[all_results.apply(_is_real_team, axis=1)]
        assigned_total = len(assigned_real)
        ok_count = (input_total == assigned_total)

        if not ok_count:
            print(f"VALIDACIÓ (COMPTATGE): Esperats {input_total} equips, assignats {assigned_total} (excloent dummies).")
            # detall per categoria
            per_cat_in = df.groupby("Nom Lliga").size().to_dict()
            per_cat_assigned = (
                assigned_real.groupby("Nom Lliga").size().to_dict()
                if not assigned_real.empty else {}
            )
            diffs = {k: (per_cat_in.get(k, 0), per_cat_assigned.get(k, 0)) for k in set(per_cat_in) | set(per_cat_assigned)}
            for cat, (exp, got) in sorted(diffs.items()):
                if exp != got:
                    print(f"  - [{cat}] esperats {exp}, assignats {got}")
        else:
            print("VALIDACIÓ (COMPTATGE): OK – totals coincideixen (excloent dummies).")

        # 2) Verificació de conflictes d'entitat dins de cada grup per categoria
        any_conflicts = False
        for cat, df_cat_res in all_results.groupby("Nom Lliga"):
            for grup, df_grup in df_cat_res.groupby("Grup"):
                ents = [e for e in df_grup["Entitat"].tolist() if e and e != "Dummy"]
                cnt = pd.Series(ents).value_counts() if ents else pd.Series(dtype=int)
                dup = cnt[cnt > 1]
                if not dup.empty:
                    any_conflicts = True
                    print(f"VALIDACIÓ (ENTITAT): Conflicte a {cat} / {grup} → {dup.to_dict()}")
        if not any_conflicts:
            print("VALIDACIÓ (ENTITAT): OK – cap grup té duplicats d'entitat.")

        # 3) Coherència de slots per entitats que han demanat CASA/FORA (mateix número arreu de categories)
        incoherencies = []
        # Prepara mapatge de slot per entitat
        preferits_casa = [8, 6, 7, 1]
        preferits_fora = [5, 4, 3, 2]

        # Ja tenim aquests diccionaris calculats abans
        # entitats_to_slot_casa, entitats_to_slot_fora

        # Helper per comprovar una entitat
        def _check_entity_consistency(entitat: str, slot_casa: int | None, slot_fora: int | None, priority_casa: bool):
            sub = all_results[all_results["Entitat"] == entitat].copy()
            if sub.empty:
                return
            # només files d'equips reals
            sub = sub[sub.apply(_is_real_team, axis=1)]
            if sub.empty:
                return
            # per cada fila, mirem el tipus de petició
            for _, r in sub.iterrows():
                req = _req_type(r.get("Núm. sorteig"))
                if req is None:
                    continue  # només validem equips que han demanat 'casa' o 'fora'
                assigned = int(r.get("Núm. sorteig assignat", 0))
                if priority_casa:
                    # Les entitats amb CASA definida: 'casa' → slot_casa, 'fora' → contrari de slot_casa
                    if req == "casa" and slot_casa is not None and assigned != slot_casa:
                        incoherencies.append((entitat, r["Nom Lliga"], r["Nom"], "casa", slot_casa, assigned))
                    if req == "fora" and slot_casa is not None and assigned != contraris.get(slot_casa):
                        incoherencies.append((entitat, r["Nom Lliga"], r["Nom"], "fora", contraris.get(slot_casa), assigned))
                else:
                    # Entitats sense CASA però amb FORA definida
                    if req == "fora" and slot_fora is not None and assigned != slot_fora:
                        incoherencies.append((entitat, r["Nom Lliga"], r["Nom"], "fora", slot_fora, assigned))
                    if req == "casa" and slot_fora is not None and assigned != contraris.get(slot_fora):
                        incoherencies.append((entitat, r["Nom Lliga"], r["Nom"], "casa", contraris.get(slot_fora), assigned))

        # Comprovem primer les entitats amb CASA (prioritat)
        for entitat, slot_casa in entitats_to_slot_casa.items():
            _check_entity_consistency(entitat, slot_casa=slot_casa, slot_fora=None, priority_casa=True)
        # Després, les entitats només amb FORA
        for entitat, slot_fora in entitats_to_slot_fora.items():
            if entitat in entitats_to_slot_casa:
                continue
            _check_entity_consistency(entitat, slot_casa=None, slot_fora=slot_fora, priority_casa=False)

        if incoherencies:
            print("VALIDACIÓ (CASA/FORA): S'han detectat incoherències d'slot per entitat:")
            for entitat, cat, nom, req, esperat, assignat in incoherencies:
                print(f"  - {entitat} · {cat} · {nom} · petició={req} → esperat {esperat}, assignat {assignat}")
        else:
            print("VALIDACIÓ (CASA/FORA): OK – cada entitat manté el mateix número per CASA i l'oposat per FORA a totes les categories.")

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