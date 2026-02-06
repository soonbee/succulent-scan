<?php

use App\Http\Controllers\AnalysisController;
use Illuminate\Support\Facades\Route;
use Inertia\Inertia;

Route::get('/', function () {
    return Inertia::render('Home/index');
})->name('home');

Route::get('/analyze', fn () => redirect()->route('home'));
Route::post('/analyze', [AnalysisController::class, 'store'])->name('analyze.store');
