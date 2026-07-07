# RFC 0001

## Title

Project Architecture

## Status

Accepted

## Goal

Create a modular Machine Learning framework written in Go.

## Principles

- Simplicity
- Performance
- Immutability
- Testability
- Zero Global State

## Layers

Public API

↓

Tensor

↓

TensorImpl

↓

Storage

↓

Buffer

↓

Backend