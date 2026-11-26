use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use crate::InferError;
use crate::item_icon::parse_csv;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeaponNameMatch {
    pub name: String,
    pub distance: usize,
    pub reliable: bool,
}

#[derive(Debug, Clone)]
pub struct WeaponNameCatalog {
    names: Vec<String>,
}

impl WeaponNameCatalog {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, InferError> {
        Self::from_csv(&fs::read_to_string(path.as_ref())?)
    }

    pub fn from_csv(csv: &str) -> Result<Self, InferError> {
        let mut records = parse_csv(csv)?
            .into_iter()
            .filter(|(_, fields)| !(fields.len() == 1 && fields[0].is_empty()));
        let (header_line, headers) = records
            .next()
            .ok_or_else(|| InferError::Runtime("ItemV2 CSV 为空或缺少表头行".into()))?;
        let column = |name: &str| -> Result<usize, InferError> {
            let matches: Vec<_> = headers
                .iter()
                .enumerate()
                .filter(|(_, header)| header.trim().eq_ignore_ascii_case(name))
                .map(|(index, _)| index)
                .collect();
            if matches.len() != 1 {
                return Err(InferError::Runtime(format!(
                    "ItemV2 CSV 第 {header_line} 行的字段 {name:?} 必须恰好出现一次"
                )));
            }
            Ok(matches[0])
        };
        let name_index = column("item_name")?;
        let class_index = column("item_class_id")?;
        let required_fields = name_index.max(class_index);
        let mut names = BTreeSet::new();
        for (line, fields) in records {
            if fields.len() <= required_fields {
                return Err(InferError::Runtime(format!(
                    "ItemV2 CSV 第 {line} 行字段数量不足"
                )));
            }
            if fields[class_index]
                .trim()
                .get(..7)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("weapon:"))
            {
                let name = fields[name_index].trim();
                if !name.is_empty() {
                    names.insert(name.to_string());
                }
            }
        }
        if names.is_empty() {
            return Err(InferError::Runtime("ItemV2 CSV 未包含武器名称记录".into()));
        }
        Ok(Self {
            names: names.into_iter().collect(),
        })
    }

    pub fn match_name(&self, ocr_text: &str) -> Result<WeaponNameMatch, InferError> {
        let normalized = normalize_weapon_name(ocr_text.trim());
        if normalized.is_empty() {
            return Err(InferError::Runtime("武器名称 OCR 结果为空".into()));
        }
        let mut candidates: Vec<_> = self
            .names
            .iter()
            .map(|name| {
                let normalized_name = normalize_weapon_name(name);
                let distance = levenshtein_distance(&normalized, &normalized_name);
                (name, normalized_name, distance)
            })
            .filter(|(_, normalized_name, _)| {
                normalized_name.chars().count() == normalized.chars().count()
            })
            .map(|(name, _, distance)| WeaponNameMatch {
                name: name.clone(),
                distance,
                reliable: false,
            })
            .collect();
        candidates.sort_by(|left, right| {
            left.distance
                .cmp(&right.distance)
                .then_with(|| left.name.cmp(&right.name))
        });
        let Some(best) = candidates.first() else {
            return Err(InferError::Runtime(
                "武器名称表中没有与 OCR 文本等长的候选".into(),
            ));
        };
        let ambiguous = candidates
            .iter()
            .filter(|candidate| candidate.distance == best.distance)
            .count()
            != 1;
        let mut result = best.clone();
        result.reliable = result.distance <= 1 && !ambiguous;
        Ok(result)
    }
}

pub fn normalize_weapon_name(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '鉄' | '鐵' => '铁',
            '黒' => '黑',
            '劍' | '剣' => '剑',
            '蝕' => '蚀',
            '鍾' | '鐘' => '钟',
            '銀' => '银',
            '獵' | '猟' => '猎',
            '彈' | '弾' => '弹',
            '釘' => '釭',
            _ => ch,
        })
        .collect()
}

fn levenshtein_distance(source: &str, target: &str) -> usize {
    let source: Vec<char> = source.chars().collect();
    let target: Vec<char> = target.chars().collect();
    let mut previous: Vec<usize> = (0..=source.len()).collect();
    let mut current = vec![0; source.len() + 1];
    for (target_index, target_ch) in target.iter().enumerate() {
        current[0] = target_index + 1;
        for (source_index, source_ch) in source.iter().enumerate() {
            let substitution = usize::from(source_ch != target_ch);
            current[source_index + 1] = (current[source_index] + 1)
                .min(previous[source_index + 1] + 1)
                .min(previous[source_index] + substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[source.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> WeaponNameCatalog {
        WeaponNameCatalog::from_csv(
            "\u{feff}item_name,item_class_id,weapon_type\n\
             西风剑,weapon:sword,sword\n\
             西风剑,weapon:sword_awaken,sword\n\
             弹弓,weapon:bow,bow\n\
             銀釘,weapon:polearm,polearm\n\
             苹果,consumable:apple,none",
        )
        .unwrap()
    }

    #[test]
    fn csv_loads_unique_weapons_and_corrects_one_character() {
        let catalog = catalog();
        let matched = catalog.match_name("西风剑").unwrap();
        assert_eq!(matched.name, "西风剑");
        assert!(matched.reliable);
        let corrected = catalog.match_name("西风创").unwrap();
        assert_eq!(corrected.name, "西风剑");
        assert_eq!(corrected.distance, 1);
        assert!(corrected.reliable);
        assert_eq!(catalog.match_name("弹弓").unwrap().name, "弹弓");
        assert_eq!(catalog.match_name("银釭").unwrap().name, "銀釘");
    }

    #[test]
    fn empty_wrong_length_and_ambiguous_names_fail_closed() {
        let catalog = catalog();
        assert!(catalog.match_name("  ").is_err());
        assert!(catalog.match_name("西风长剑").is_err());
        assert!(catalog.match_name("完全不相关的名字").is_err());

        let ambiguous = WeaponNameCatalog::from_csv(
            "item_name,item_class_id\n西风剑,weapon:sword\n西风创,weapon:sword\n",
        )
        .unwrap();
        assert!(!ambiguous.match_name("西风锻").unwrap().reliable);
    }

    #[test]
    fn malformed_or_weapon_free_csv_is_rejected() {
        for csv in [
            "item_name,item_class_id\n苹果,consumable:apple",
            "item_name,item_class_id\n西风剑",
            "item_name,item_class_id\n,weapon:sword",
            "item_name,item_name,item_class_id\n西风剑,西风剑,weapon:sword",
            "item_name,item_class_id\n\"西风剑,weapon:sword",
        ] {
            assert!(WeaponNameCatalog::from_csv(csv).is_err(), "{csv}");
        }
    }
}
