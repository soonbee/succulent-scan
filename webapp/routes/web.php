<?php

use App\Http\Controllers\AnalysisController;
use App\Http\Controllers\DescriptionController;
use Illuminate\Support\Facades\Route;
use Inertia\Inertia;

Route::get('/', function () {
    return Inertia::render('Home/index');
})->name('home');

Route::get('/analysis', fn () => Inertia::render('Analysis'))->name('analysis.index');
Route::post('/analysis', [AnalysisController::class, 'store'])->name('analysis.store');

Route::get('/description', [DescriptionController::class, 'show'])->name('description');
