#!/bin/bash
set -x

echo "Module 0"
pytest tests -vv -m task0_1
pytest tests -vv -m task0_2
pytest tests -vv -m task0_3
pytest tests -vv -m task0_4

echo "Module 1"
pytest tests -vv -m task1_1
pytest tests -vv -m task1_2
pytest tests -vv -m task1_3
pytest tests -vv -m task1_4

echo "Module 2"
pytest tests -vv -m task2_1
pytest tests -vv -m task2_2
pytest tests -vv -m task2_3
pytest tests -vv -m task2_4

echo "Module 3"
pytest tests -vv -m task3_1
pytest tests -vv -m task3_2
pytest tests -vv -m task3_3
pytest tests -vv -m task3_4

echo "Module 4"
pytest tests -vv -m task4_1
pytest tests -vv -m task4_2
pytest tests -vv -m task4_3
pytest tests -vv -m task4_4
