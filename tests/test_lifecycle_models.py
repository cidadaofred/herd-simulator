import unittest

from src.simulator.herd_nutrition import HerdNutritionModel
from src.simulator.herd_social import HerdSocialModel


class LifecycleModelTests(unittest.TestCase):
    def test_newborn_enters_nutrition_and_mothers_lot(self):
        nutrition = HerdNutritionModel(
            {
                "categories": {
                    "terneiro": {
                        "count": 1,
                        "cohort_count": 1,
                        "daily_intake_target_grams": 4500,
                    },
                    "vaca_lactante": {
                        "count": 1,
                        "cohort_count": 1,
                        "daily_intake_target_grams": 11000,
                    },
                }
            },
            2,
        )
        social = HerdSocialModel(
            {
                "integration_days": 6,
                "separate_shelter_until_familiarity": 0.67,
                "new_lot_separation_px": 125,
                "established_lot_separation_px": 28,
                "mother_attraction": 0.58,
                "lots": [
                    {
                        "id": "lote_base",
                        "animal_ids": "1-2",
                        "arrival_day": 1,
                        "preferred_shelter": "P1",
                        "enabled": True,
                    }
                ],
            },
            2,
        )

        newborn = nutrition.add_newborn(3)
        social.add_newborn(3, 2)

        self.assertEqual(newborn.category, "terneiro")
        self.assertEqual(social.lot_for(3), "lote_base")
        self.assertEqual(social.mother_of(3), 2)
        self.assertIn(3, social.active_ids(1))


if __name__ == "__main__":
    unittest.main()
