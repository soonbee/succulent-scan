<?php

namespace App\Http\Controllers;

use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Inertia\Inertia;
use Inertia\Response;

class DescriptionController extends Controller
{
    private const SUPPORTED_GENERA = [
        'echeveria',
        'haworthia',
        'aeonium',
        'dudleya',
        'graptopetalum',
        'lithops',
        'pachyphytum',
    ];

    public function show(Request $request): Response|RedirectResponse
    {
        $genus = $request->query('genus');

        if (! $genus || ! in_array(strtolower($genus), self::SUPPORTED_GENERA, true)) {
            return redirect()->route('home');
        }

        return Inertia::render('Description', [
            'genus' => strtolower($genus),
        ]);
    }
}
