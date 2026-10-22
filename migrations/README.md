# intracellular-abfdir normalization

One-off scripts that clean up the registry metadata of the
`intracellular-abfdir` resources and then migrate them to
`intracellular-abfzip`. The migration copies each record's metadata into the
zip's `metadata.json`, where the hash fixes it, so it has to be right first.
Deleted once the migration is done.

## How it runs

1. `normalize_abfdir.py` reads the registry, the archive and the bird
   database, and writes a plan. It changes nothing.

   ```
   uv run python migrations/normalize_abfdir.py \
       --archive <intracellular archive root> \
       --names migrations/abfdir-names.tsv \
       --animals migrations/abfdir-animals.tsv \
       --accept migrations/abfdir-accept.tsv \
       -o abfdir-changes.jsonl --table abfdir-names.tsv
   ```

   - `abfdir-changes.jsonl`: one line per resource, with the `update` to send
     (`null` deletes a key), the record as it was (`before`), anything left for
     `review`, and `notes` on how decisions were made.
   - `abfdir-names.tsv` (the `--table`): the evidence for each resource's name
     and animal, including its files, ABF dates, Clampex file numbers and
     protocols. For review only.

2. `apply_abfdir_changes.py` sends the updates. It applies only records with
   nothing left for review, skips any record that changed since the plan or is
   already done (so it can be rerun), checks each record the registry returns,
   and stops at the first problem. `--dry-run` checks and changes nothing.

   ```
   uv run python migrations/apply_abfdir_changes.py --dry-run abfdir-changes.jsonl
   uv run python migrations/apply_abfdir_changes.py --log abfdir-applied.jsonl abfdir-changes.jsonl
   ```

   The log has every record before and after; keep it.

3. `abfdir_to_abfzip.py` migrates the normalized resources, one archive at a
   time. For each resource it builds `<id>.zip` in the archive's staging
   directory with `abf-zip`'s builder: metadata from the registry record plus
   the ABF files' `software` and `processed_by`
   (`org.meliza.melizalab-tools/abfdir-to-abfzip <version>`), and a catalog
   record with its SHA-1 for every file. It checks the zip (the reader, the
   metadata, each member against its hash, and that the files' hashes give the
   directory's registered hash, which ties the zip to the resource it
   replaces), stores it, changes the resource's sha1 and dtype and adds the two
   metadata keys in the registry, and removes the directory last. Resources
   that fail a check are skipped and logged; an error while storing, updating
   the registry or removing stops the run, and running it again finishes what
   was interrupted.

   nbank 0.8 to 0.10 (2023-02-13 to 2025-07-03) hashed directories by their
   absolute paths, so the 485 resources deposited then don't match their
   hashes. `diagnose_abfdir_hashes.py` found the directory 240 of them were
   deposited from (bellanger's `/archive/intracellular/<recording>`,
   rallapalli's `/home/mtm7ah/staging/intracellular/<recording>`), and the
   migration checks those hashes; the other 245 (all yl6tm's, and seven of
   bellanger's from 2025-04-29 to 05-01) are migrated unverified. No other
   resource's hash differs, and no file in any of them changed after it was
   registered. The results are in `abfdir-hashes.tsv`, and the log records how
   each zip's hash was checked.

   ```
   nbank check archive --fix <root>          # creates staging/ in older archives
   uv run python migrations/abfdir_to_abfzip.py --dry-run --hashes migrations/abfdir-hashes.tsv <root>
   uv run python migrations/abfdir_to_abfzip.py --hashes migrations/abfdir-hashes.tsv --log abfzip-migrated.jsonl <root>
   nbank check all <root>
   ```

   Changing a resource's sha1 needs a registry superuser, and the account
   running it needs write access to the archive. `--limit N` and a list of ids
   migrate a few first.

## Correction files

Decisions the script can't make are in three TSVs, each with a `why` column:

- `abfdir-names.tsv`: recording names, by id.
- `abfdir-animals.tsv`: animals, by id.
- `abfdir-accept.tsv`: records reviewed and accepted as they are, with an
  optional `comment` set in their metadata.

## The changes

Examples are from the plan of 2026-10-21 (ids shortened).

### `name` becomes `recording` (all records)

The form is `YYYYMMDD_SLICE_CELL`. 788 names deposited since 2022 were read by
`nbank -k` as integers, because Python allows underscores in numbers
(`20140603_1_1` became 2014060311). They're recovered where only one reading
fits; the rest were settled by the ABF files' recording dates, the order of
Clampex's file numbers within a day, or a correction file.

| id       | `name`           | `recording`     | how                                                     |
|----------|------------------|-----------------|---------------------------------------------------------|
| 00358d50 | `"20160526_1_5"` | `20160526_1_5`  | kept                                                    |
| 0064fea6 | `2025111332`     | `20251113_3_2`  | recovered                                               |
| 1a8808cd | `202450614`      | `20240506_1_4`  | recovered (the only valid date)                         |
| 0594a21a | `202461114`      | `20240611_1_4`  | the ABF files' date                                     |
| 6519d899 | `20240520111`    | `20240520_1_11` | given: its files follow `1_10`'s                        |
| 28f142ad | `2021051823`     | `20210518_2_4`  | given: a second `2_3`, numbered between `2_3` and `2_5` |
| dcb43187 | `"20210102_2_2"` | `20210101_2_2`  | given: its ABF files and images are from 2021-01-01     |
| a126271a | (none)           | `20181129_1_6`  | given: its files fall between `1_5` and `1_7`           |

### Animals and experimenters

The bird database's event log says which animal was used for slices on each
recording date. Records with no animal get that one; records whose animal
wasn't used that day were corrected (`abfdir-animals.tsv`), usually because a
recycled command carried an earlier deposit's animal forward. The 284 records
from 2018 without an experimenter are anc4kj's.

| id       | recording      | before                            | after                                   |
|----------|----------------|-----------------------------------|-----------------------------------------|
| 057bd896 | `20140708_1_1` | no experimenter or animal         | anc4kj; fb16519d (used that day)        |
| fe0359c2 | `20170110_1_1` | a164c97c (used 01-11)             | 7f517da7 (used 01-10)                   |
| 0e9a621b | `20170419_1_1` | bbae12f0 (red_199, lived to 2018) | ef8b1843, its nestmate (used 04-19)     |
| cb37f71e | `20170421_1_2` | c96d6801 (red_200, lived to 2018) | 46e3c26d (red_201, same clutch and age) |
| a126271a | `20181129_1_6` | 25b99af0 (hatched 2019)           | 1ba8fcb4 (the session's other cells)    |

`biid`, an animal id entered by hand, is deleted; where it disagreed with the
animal (3bd85723 and three others), the event log confirmed the animal.

### `location` (all records)

The region recorded from: CM for anc4kj, yl6tm, bellanger and dmeliza; NCM
for rallapalli, or dNCM/vNCM where he noted `region: dorsal` or `ventral`.

| id       | before                                    | after                                                             |
|----------|-------------------------------------------|-------------------------------------------------------------------|
| 00358d50 | (none)                                    | `location: CM`                                                    |
| 0064fea6 | (none)                                    | `location: NCM`                                                   |
| 07e49824 | `region: dorsal`                          | `location: dNCM`                                                  |
| e4052cd1 | `location: "uncertain; may not be in CM"` | `location: CM`, `comment: "location uncertain; may not be in CM"` |

### Other keys

| id       | before                                     | after                                  |
|----------|--------------------------------------------|----------------------------------------|
| 28f142ad | `filled: "biocytin"`                       | `biocytin: true` (387 records)         |
| 07e49824 | `rearing: "colony"`                        | `condition: "colony"` (8)              |
| 2f8bc4ff | `hemisphere: "right"`, `slice: "saggital"` | `hemisphere: "R"`, `slice: "sagittal"` |
| 46f5857b | `hemisphere: "unknown"`                    | (deleted)                              |

### Accepted as they are (`abfdir-accept.tsv`)

| id       | recording      | why                                                | comment set                                   |
|----------|----------------|----------------------------------------------------|-----------------------------------------------|
| 773cb484 | `20191206_1_1` | the rig's clock read 2014-01-10                    |                                               |
| 85b3eab4 | `20241114_2_5` | one file, from five days after its animal was used | animal and name uncertain                     |
| bad08eac | `20250719_2_2` | files from two sessions (to be split)              |                                               |
| 79d20768 | `20250128_1_1` | the cell this name refers to                       |                                               |
| 8926aee2 | `20250128_1_1` | other recordings deposited under the same name     | not cell `20250128_1_1`; actual cells unknown |

bellanger's other two pairs (`20250129_1_1`, `20250129_1_2`) are handled like
79d20768 and 8926aee2.
