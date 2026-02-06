<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Inertia\Inertia;
use Inertia\Response;

class AnalysisController extends Controller
{
    public function store(Request $request): Response
    {
        $request->validate([
            'image' => 'required|image|max:10240',
        ]);

        sleep(3);

        $results = [
            ['ko' => '에오니움', 'en' => 'aeonium', 'acc' => 78],
            ['ko' => '에케베리아', 'en' => 'echeveria', 'acc' => 48],
            ['ko' => '리톱스', 'en' => 'lithops', 'acc' => 13],
        ];

        return Inertia::render('Result', [
            'results' => $results,
        ]);
    }
}
